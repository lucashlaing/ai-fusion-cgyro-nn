"""
Compare per-target distributions across two or more h5 dataset folders.

For each of the 4 output targets (G_elec, Q_elec, Q_ions, P_ions), produces
one PNG containing a subplot per input folder, all sharing the same x-axis
range so distributions can be compared directly.

Targets are derived from `sumf` using the same convention as the dataset
loader (Spectra_Regularization._read_path):
    sumf[:, :, 0, :, :, :]  ->  sum over nf axis  ->  (N, nky, ns, 5)
        G_elec   = [:, :, 0, 0]
        Q_elec   = [:, :, 0, 1]
        Q_ions   = sum([:, :, 1:, 1])
        P_ions   = sum([:, :, 1:, 2])

Usage:
    python test/plot_target_distributions.py <folder1> <folder2> [--labels L1 L2] [--name comparison]
"""
import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_NAMES = ["G_elec", "Q_elec", "Q_ions", "P_ions"]
COLORS = {"G_elec": "tab:blue", "Q_elec": "tab:orange",
          "Q_ions": "tab:green", "P_ions": "tab:red"}


def derive_targets(sumf):
    flux = sumf[:, :, 0, :, :, :]               # (N, nky, nf, ns, 5)
    flux = np.sum(flux, axis=2)                 # (N, nky, ns, 5)
    G_e = flux[:, :, 0, 0]
    Q_e = flux[:, :, 0, 1]
    Q_i = np.sum(flux[:, :, 1:, 1], axis=-1)
    P_i = np.sum(flux[:, :, 1:, 2], axis=-1)
    return {"G_elec": G_e, "Q_elec": Q_e, "Q_ions": Q_i, "P_ions": P_i}


def load_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "**/*.h5"), recursive=True))
    print(f"  Found {len(files)} h5 files under {folder}")
    pooled = {n: [] for n in TARGET_NAMES}
    ky_pool = []
    for path in files:
        with h5py.File(path, "r") as f:
            sumf = np.asarray(f["sumf"])
            ky = np.asarray(f["ky"])
            mask = np.asarray(f["meta/failed_mask"]).astype(bool) if "meta/failed_mask" in f else None
        targets = derive_targets(sumf)
        N = min(ky.shape[0], next(iter(targets.values())).shape[0])
        if mask is not None:
            N = min(N, mask.shape[0])
        ky = ky[:N]
        targets = {k: v[:N] for k, v in targets.items()}
        good = (~mask[:N]) if mask is not None else np.ones(targets["G_elec"].shape, dtype=bool)
        for name, arr in targets.items():
            valid = good & np.isfinite(arr)
            pooled[name].append(arr[valid])
        # ky values: drop masked slots, drop zero-padding (cgyro pads trailing kys with 0)
        ky_valid = good & np.isfinite(ky) & (ky > 0)
        ky_pool.append(ky[ky_valid])
    pooled = {name: np.concatenate(parts) for name, parts in pooled.items()}
    pooled["__ky__"] = np.concatenate(ky_pool) if ky_pool else np.array([])
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="folders containing .h5 files (recursive)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display label per folder (defaults to folder basename)")
    ap.add_argument("--name", default="compare",
                    help="output prefix (PNGs land at test/plots/<name>_<target>.png)")
    ap.add_argument("--clip-percentile", type=float, default=99.0,
                    help="x-axis clipped to [100-p, p] of each dataset; widest is taken (default 99)")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.folders):
        raise ValueError("--labels count must match number of folders")
    labels = args.labels or [os.path.basename(os.path.normpath(f)) for f in args.folders]

    print("Loading datasets:")
    data = []
    for folder, label in zip(args.folders, labels):
        print(f"\n[{label}]")
        targets = load_folder(folder)
        data.append((label, targets))
        for name in TARGET_NAMES:
            arr = targets[name]
            print(f"    {name:7s}  N={len(arr):>10d}  mean={arr.mean():>11.4g}  "
                  f"std={arr.std():>11.4g}  range=[{arr.min():>11.4g}, {arr.max():>11.4g}]")
        ky = targets["__ky__"]
        if ky.size:
            print(f"    {'ky':7s}  N={len(ky):>10d}  mean={ky.mean():>11.4g}  "
                  f"std={ky.std():>11.4g}  range=[{ky.min():>11.4g}, {ky.max():>11.4g}]")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "test", "plots")
    os.makedirs(out_dir, exist_ok=True)

    p_lo, p_hi = 100 - args.clip_percentile, args.clip_percentile

    for tname in TARGET_NAMES:
        # Shared x-range: union of per-dataset [p_lo, p_hi] so each bulk is visible.
        los = [np.percentile(d[1][tname], p_lo) for d in data]
        his = [np.percentile(d[1][tname], p_hi) for d in data]
        x_lo, x_hi = min(los), max(his)

        fig, axes = plt.subplots(len(data), 1, figsize=(10, 3.5 * len(data)), sharex=True)
        if len(data) == 1:
            axes = [axes]
        for ax, (label, targets) in zip(axes, data):
            arr = targets[tname]
            visible = arr[(arr >= x_lo) & (arr <= x_hi)]
            n_clipped = len(arr) - len(visible)
            ax.hist(visible, bins=120, alpha=0.8, density=True, color=COLORS[tname])
            ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.2,
                       label=f"mean={arr.mean():.3g}")
            ax.set_xlim(x_lo, x_hi)
            ax.set_yscale("log")
            ax.set_ylabel("density (log)")
            ax.set_title(f"{label}   N={len(arr)}   σ={arr.std():.3g}   "
                         f"range=[{arr.min():.3g}, {arr.max():.3g}]   "
                         f"({n_clipped} pts outside view)")
            ax.legend(fontsize=9)
        axes[-1].set_xlabel(
            f"{tname} value (x ∈ [{x_lo:.3g}, {x_hi:.3g}] = "
            f"union of {p_lo:g}th-{p_hi:g}th percentiles across datasets)"
        )
        fig.suptitle(f"{tname} — distribution comparison", fontsize=12)
        plt.tight_layout()
        out = os.path.join(out_dir, f"{args.name}_{tname}.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nSaved: {out}")

        # Per-dataset full-range plot for this target (no clip).
        for label, targets in data:
            arr = targets[tname]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(arr, bins=200, alpha=0.8, density=True, color=COLORS[tname])
            ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.2,
                       label=f"mean={arr.mean():.3g}")
            ax.set_yscale("log")
            ax.set_ylabel("density (log)")
            ax.set_xlabel(tname)
            ax.set_title(f"{label} {tname} — full range   N={len(arr)}   "
                         f"σ={arr.std():.3g}   "
                         f"range=[{arr.min():.3g}, {arr.max():.3g}]")
            ax.legend(fontsize=9)
            plt.tight_layout()
            out = os.path.join(out_dir, f"{args.name}_{tname}_{label}_full.png")
            plt.savefig(out, dpi=120)
            plt.close(fig)
            print(f"Saved: {out}")

    # ky comparison: stacked subplots (same style as targets), shared x range, log-y.
    ky_data = [(label, targets["__ky__"]) for label, targets in data
               if targets["__ky__"].size > 0]
    if ky_data:
        los = [np.percentile(arr, p_lo) for _, arr in ky_data]
        his = [np.percentile(arr, p_hi) for _, arr in ky_data]
        x_lo, x_hi = min(los), max(his)

        fig, axes = plt.subplots(len(ky_data), 1, figsize=(10, 3.5 * len(ky_data)),
                                 sharex=True)
        if len(ky_data) == 1:
            axes = [axes]
        for ax, (label, arr) in zip(axes, ky_data):
            visible = arr[(arr >= x_lo) & (arr <= x_hi)]
            n_clipped = len(arr) - len(visible)
            ax.hist(visible, bins=120, alpha=0.8, density=True, color="tab:purple")
            ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.2,
                       label=f"mean={arr.mean():.3g}")
            ax.set_xlim(x_lo, x_hi)
            ax.set_yscale("log")
            ax.set_ylabel("density (log)")
            ax.set_title(f"{label}   N={len(arr)}   σ={arr.std():.3g}   "
                         f"range=[{arr.min():.3g}, {arr.max():.3g}]   "
                         f"({n_clipped} pts outside view)")
            ax.legend(fontsize=9)
        axes[-1].set_xlabel(
            f"ky  (x ∈ [{x_lo:.3g}, {x_hi:.3g}] = "
            f"union of {p_lo:g}th-{p_hi:g}th percentiles across datasets)"
        )
        fig.suptitle("ky — distribution comparison", fontsize=12)
        plt.tight_layout()
        out = os.path.join(out_dir, f"{args.name}_ky.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nSaved: {out}")

    # Per-dataset full-range ky plot (no shared scale, no percentile clip).
    for label, arr in ky_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(arr, bins=200, alpha=0.8, density=True, color="tab:purple")
        ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={arr.mean():.3g}")
        ax.set_yscale("log")
        ax.set_ylabel("density (log)")
        ax.set_xlabel("ky")
        ax.set_title(f"{label} ky — full range   N={len(arr)}   "
                     f"σ={arr.std():.3g}   "
                     f"range=[{arr.min():.3g}, {arr.max():.3g}]")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(out_dir, f"{args.name}_ky_{label}_full.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
