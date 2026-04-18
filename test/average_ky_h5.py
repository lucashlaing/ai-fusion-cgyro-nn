"""
Two-pass global ky averaging across all h5 files in a folder.

Pass 1: for every `ky` dataset (shape (N, nky)), accumulate per-column
sum and count across ALL files. This gives one mean vector of length nky
per unique dataset path.

Pass 2: rewrite each file, replacing each `ky` dataset with the global
per-column mean vector broadcast back to (N, nky). Every other dataset is
copied unchanged.
"""

import argparse
import glob
import os

import h5py
import numpy as np

KY_KEY = "ky"


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def walk_ky_datasets(h5_obj, prefix=""):
    """Yield (full_path, dataset) for every `ky` dataset in the file."""
    for name, item in h5_obj.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            yield from walk_ky_datasets(item, path)
        elif isinstance(item, h5py.Dataset) and name == KY_KEY:
            yield path, item


def accumulate_ky(h5_files):
    """Pass 1: compute per-column sum and count for every unique ky path."""
    sums = {}
    counts = {}
    for src_path in h5_files:
        with h5py.File(src_path, "r") as f:
            for path, ds in walk_ky_datasets(f):
                arr = ds[()]
                if not (isinstance(arr, np.ndarray) and arr.ndim >= 2):
                    continue
                mask = ~np.isnan(arr)
                contrib_sum = np.where(mask, arr, 0.0).sum(axis=0)
                contrib_cnt = mask.sum(axis=0)
                if path not in sums:
                    sums[path] = contrib_sum.astype(np.float64)
                    counts[path] = contrib_cnt.astype(np.int64)
                else:
                    sums[path] += contrib_sum
                    counts[path] += contrib_cnt
    means = {}
    for path in sums:
        with np.errstate(invalid="ignore", divide="ignore"):
            means[path] = np.where(counts[path] > 0, sums[path] / counts[path], np.nan)
    return means


def process_item(name, src_item, dst_parent, full_path, ky_means):
    if isinstance(src_item, h5py.Group):
        grp = dst_parent.create_group(name)
        copy_attrs(src_item, grp)
        for sub_name, sub_item in src_item.items():
            process_item(sub_name, sub_item, grp, f"{full_path}/{sub_name}", ky_means)
    elif isinstance(src_item, h5py.Dataset):
        data = src_item[()]
        if (
            name == KY_KEY
            and isinstance(data, np.ndarray)
            and data.ndim >= 2
            and full_path in ky_means
        ):
            mean_vec = ky_means[full_path]  # shape (nky,)
            data = np.broadcast_to(mean_vec[None, :], data.shape).astype(data.dtype, copy=True)
        ds = dst_parent.create_dataset(name, data=data)
        copy_attrs(src_item, ds)


def process_file(src_path: str, dst_path: str, ky_means) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        copy_attrs(fin, fout)
        for name, item in fin.items():
            process_item(name, item, fout, name, ky_means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input data folder containing .h5 files")
    parser.add_argument("-o", "--output", required=True, help="Output folder for averaged .h5 files")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)

    h5_files = sorted(glob.glob(os.path.join(in_dir, "**", "*.h5"), recursive=True))
    if not h5_files:
        print(f"No .h5 files found in {in_dir}")
        return

    print(f"Found {len(h5_files)} h5 files in {in_dir}")

    print("Pass 1: accumulating per-column ky sums across all files...")
    ky_means = accumulate_ky(h5_files)
    for path, mean_vec in ky_means.items():
        print(f"  {path}: nky={mean_vec.shape[0]}, mean[0..3]={mean_vec[:3]}")

    print("Pass 2: writing output files with ky replaced by global per-column mean...")
    for src_path in h5_files:
        rel = os.path.relpath(src_path, in_dir)
        dst_path = os.path.join(out_dir, rel)
        print(f"  {rel} -> {dst_path}")
        process_file(src_path, dst_path, ky_means)

    print(f"Done. Wrote averaged files to {out_dir}")


if __name__ == "__main__":
    main()
