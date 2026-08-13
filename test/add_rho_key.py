"""Add a discrete `rho` key (canonical labels 0.1..0.9) to the h5s by snapping
RMIN_LOC to its nearest radial peak. Mutates pool/ and test/ IN-PLACE.
"""
import glob
import os
import h5py
import numpy as np

ROOT = "/data/lucas_work/tglf_sumf_data_major_minor_perturb_madcut_filter"
DIRS = ["pool", "test"]

# 9 radial peaks found by scanning RMIN_LOC -> canonical rho labels 0.1..0.9
PEAKS  = np.array([0.114, 0.231, 0.348, 0.463, 0.570, 0.670, 0.766, 0.857, 0.933])
LABELS = np.array([0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9])


def rho_from_rmin(rmin):
    idx = np.abs(rmin.reshape(-1, 1) - PEAKS[None, :]).argmin(axis=1)
    return LABELS[idx].astype(np.float64), idx


for d in DIRS:
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.h5"))):
        with h5py.File(f, "a") as h:
            rmin = np.asarray(h["RMIN_LOC"][:])
            rho, idx = rho_from_rmin(rmin)
            rho = rho.reshape(h["RMIN_LOC"].shape)   # match RMIN_LOC shape
            if "rho" in h:
                del h["rho"]
            h.create_dataset("rho", data=rho)
            worst = np.abs(rmin.ravel() - PEAKS[idx]).max()
            counts = np.bincount(idx, minlength=9)
        print(f"[{d}] {os.path.basename(f)[:42]:42} N={rmin.size:>8}  "
              f"max_snap={worst:.4f}  counts={counts.tolist()}")

print("\nDone. rho added to pool/ and test/.")
