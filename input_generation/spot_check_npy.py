#!/usr/bin/env python3
# print_random_points.py
import numpy as np

INPUT_KEYS = [
    "RLTS_3","KAPPA_LOC","ZETA_LOC","TAUS_3","VPAR_1","Q_LOC","RLNS_1","TAUS_2",
    "Q_PRIME_LOC","P_PRIME_LOC","ZMAJ_LOC","VPAR_SHEAR_1","RLTS_2","S_DELTA_LOC",
    "RLTS_1","RMIN_LOC","DRMAJDX_LOC","AS_3","RLNS_3","DZMAJDX_LOC","DELTA_LOC",
    "S_KAPPA_LOC","ZEFF","VEXB_SHEAR","RMAJ_LOC","AS_2","RLNS_2","S_ZETA_LOC",
    "BETAE_log10","XNUE_log10","DEBYE_log10"
]

def print_random_points(npy_path: str, n_points: int = 10, seed: int = 42):
    arr = np.load(npy_path)
    print(f"Loaded array with shape {arr.shape}")

    rng = np.random.default_rng(seed)
    idxs = rng.choice(arr.shape[0], size=min(n_points, arr.shape[0]), replace=False)

    for i, idx in enumerate(idxs, start=1):
        print(f"\n🔹 Sample {i} (row {idx}):")
        row = arr[idx]
        for k, v in zip(INPUT_KEYS, row):
            print(f"  {k:15s}: {v:.6g}")

if __name__ == "__main__":
    NPY_FILE = "input_generation/samples_10k_minmax_normal.npy"
    print_random_points(NPY_FILE, n_points=10, seed=123)
