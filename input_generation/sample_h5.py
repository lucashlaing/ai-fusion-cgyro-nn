#!/usr/bin/env python3
# sample_h5_to_h5.py
import argparse, os, sys
import numpy as np
import h5py

def sample_indices(n_samples: int, k: int, seed: int | None):
    rng = np.random.default_rng(seed)
    k = min(k, n_samples)
    return np.sort(rng.choice(n_samples, size=k, replace=False))

def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v

def subset_copy_recursive(src_grp: h5py.Group, dst_grp: h5py.Group, idx: np.ndarray, n_samples: int,
                          compress: bool = True):
    """
    Recursively copy groups/datasets from src_grp into dst_grp.
    - If a dataset has first dimension == n_samples, subset rows using idx.
    - Otherwise, copy dataset as-is.
    - Copy all attributes.
    """
    for name, obj in src_grp.items():
        if isinstance(obj, h5py.Group):
            new_grp = dst_grp.create_group(name)
            copy_attrs(obj, new_grp)
            subset_copy_recursive(obj, new_grp, idx, n_samples, compress)
        elif isinstance(obj, h5py.Dataset):
            # Decide whether to subset along axis 0
            if obj.ndim >= 1 and obj.shape[0] == n_samples:
                sel = (idx,) + tuple(slice(None) for _ in range(obj.ndim - 1))
                data = obj[sel]
            else:
                data = obj[()]

            kwargs = {}
            if compress and data.size > 1024:  # avoid overhead for tiny arrays
                kwargs.update(dict(compression="gzip", compression_opts=4, shuffle=True))

            dset = dst_grp.create_dataset(name, data=data, dtype=obj.dtype, **kwargs)
            copy_attrs(obj, dset)
        else:
            # Unknown object type; skip or handle as needed
            pass

def main():
    ap = argparse.ArgumentParser(description="Sample K rows from an HDF5 with dataset 'ky' and write a subset HDF5.")
    ap.add_argument("--src", required=True, help="Source HDF5 path (must contain top-level dataset 'ky').")
    ap.add_argument("--dst", required=True, help="Destination HDF5 path to write the sampled subset.")
    ap.add_argument("--k", type=int, default=200, help="Number of rows to sample (default: 50).")
    ap.add_argument("--seed", type=int, default=52, help="Random seed for reproducible sampling.")
    ap.add_argument("--no-compress", action="store_true", help="Disable gzip compression on destination datasets.")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"File not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(args.src, "r") as src:
        if "ky" not in src or not isinstance(src["ky"], h5py.Dataset):
            print("Error: source HDF5 must contain a top-level dataset named 'ky'.", file=sys.stderr)
            sys.exit(1)
        ky = src["ky"]
        if ky.ndim != 2:
            print(f"Error: 'ky' must be 2D, got shape {ky.shape}.", file=sys.stderr)
            sys.exit(1)

        n_samples = ky.shape[0]
        if n_samples == 0:
            print("Error: 'ky' has zero rows; nothing to sample.", file=sys.stderr)
            sys.exit(1)

        idx = sample_indices(n_samples, args.k, args.seed)
        print(f"Sampling {idx.size} / {n_samples} rows. First few indices: {idx[:10].tolist()}")

        # Write destination file
        with h5py.File(args.dst, "w") as dst:
            # File-level attrs
            copy_attrs(src, dst)
            dst.attrs["source_file"] = os.path.abspath(args.src)
            dst.attrs["subset_count"] = int(idx.size)
            if args.seed is not None:
                dst.attrs["subset_seed"] = int(args.seed)

            # Copy everything, subsetting row-aligned datasets
            subset_copy_recursive(src, dst, idx, n_samples, compress=(not args.no_compress))

            # Also store the sampled row indices for traceability
            dst.create_dataset("sampled_idx", data=idx.astype(np.int64))

    print(f"Wrote subset HDF5: {args.dst}")

if __name__ == "__main__":
    main()
