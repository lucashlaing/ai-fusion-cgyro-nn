# ky_spectrum.py
import os
import argparse
import numpy as np
from math import sqrt, log, exp, floor
from typing import List, Dict, Any, Tuple
import h5py

# ------- Your 31-key ordering (column -> name) -------
INPUT_KEYS = [
    "RLTS_3","KAPPA_LOC","ZETA_LOC","TAUS_3","VPAR_1","Q_LOC","RLNS_1","TAUS_2",
    "Q_PRIME_LOC","P_PRIME_LOC","ZMAJ_LOC","VPAR_SHEAR_1","RLTS_2","S_DELTA_LOC",
    "RLTS_1","RMIN_LOC","DRMAJDX_LOC","AS_3","RLNS_3","DZMAJDX_LOC","DELTA_LOC",
    "S_KAPPA_LOC","ZEFF","VEXB_SHEAR","RMAJ_LOC","AS_2","RLNS_2","S_ZETA_LOC",
    "BETAE_log10","XNUE_log10","DEBYE_log10"
]
KEY_TO_COL = {k: i for i, k in enumerate(INPUT_KEYS)}

# ------- Your hard-coded global settings -------
GLOBAL_CONST: Dict[str, Any] = {
    "AS_1":  +1.0,
    "TAUS_1":+1.0,
    "MASS_1":0.0002723125672605524,
    "ZS_1":  -1.0,
    "MASS_2":+1.0,
    "ZS_2":  +1.0,
    "MASS_3":+6.0,
    "ZS_3":  +6.0,
    "UNITS": "CGYRO",          # not "GYRO" -> ky_factor = grad_r0
    "NKY":   12,
    "KYGRID_MODEL": 4,
    "USE_AVE_ION_GRID": True,
    "NS": 3
}

# ------- Minimal Julia-to-Python port (dict-based, no dataclass) -------
def get_ky_spectrum(inputs: Dict[str, Any], grad_r0: float) -> List[float]:
    """
    Required keys in `inputs`: UNITS, NKY, KYGRID_MODEL, TAUS(list), MASS(list), ZS(list), AS(list),
                               NS(int), USE_AVE_ION_GRID(bool)
    Returns: list[float] ky_spectrum
    """
    units_in = inputs["UNITS"]
    nky_in = int(inputs["NKY"])
    spectrum_type = int(inputs["KYGRID_MODEL"])

    # electrons are index 0
    rho_e = sqrt(inputs["TAUS"][0] * inputs["MASS"][0]) / abs(inputs["ZS"][0])

    rho_ion = 0.0
    charge = 0.0
    # species loop over 1..NS-1 (ions)
    for is_idx in range(1, inputs["NS"]):
        if not inputs["USE_AVE_ION_GRID"]:
            rho_ion = sqrt(inputs["TAUS"][1] * inputs["MASS"][1]) / abs(inputs["ZS"][1])
            break
        else:
            denom = abs(inputs["ZS"][0] * inputs["AS"][0])
            if denom != 0:
                if (inputs["ZS"][is_idx] * inputs["AS"][is_idx]) / denom > 0.1:
                    rho_ion += inputs["AS"][is_idx] * sqrt(inputs["TAUS"][is_idx] * inputs["MASS"][is_idx])
                    charge += inputs["ZS"][is_idx] * inputs["AS"][is_idx]
    if charge != 0.0:
        rho_ion = rho_ion / charge

    if rho_ion <= 0.0:
        raise ValueError("Computed rho_ion <= 0 (check AS/TAUS/ZS inputs).")

    ky_min = 0.05
    ky_max = 0.7
    ky_in  = 0.3

    ky_factor = 1.0 if (units_in == "GYRO") else float(grad_r0)

    ky0 = ky_min
    ky1 = ky_max
    nk_zones = 3
    if nk_zones >= 2:
        ky1 = ky_max / sqrt(inputs["MASS"][0])

    # Branches
    if spectrum_type == 0:
        nky = nky_in
        ky_spectrum = [0.0] * nky
        dky_spectrum = [0.0] * nky
        ky1 = ky_in
        dky0 = ky1 / nky_in
        for i in range(nky):
            ky_spectrum[i] = (i + 1) * dky0
            dky_spectrum[i] = dky0

    elif spectrum_type == 1:
        nky = 9
        ky_spectrum = [0.0] * (nky + nky_in)
        dky_spectrum = [0.0] * (nky + nky_in)

        ky_max_local = 0.9 * ky_factor / rho_ion
        dky0 = ky_max_local / nky
        for i in range(nky):
            ky_spectrum[i] = (i + 1) * dky0
            dky_spectrum[i] = dky0

        ky0 = ky_max_local + dky0
        ky1 = 0.4 * ky_factor / rho_e

        if nky_in > 0:
            dky0 = log(ky1 / ky0) / (nky_in - 1)
            lnky = log(ky0)
            for i in range(nky, nky + nky_in):
                ky_spectrum[i] = exp(lnky)
                dky_spectrum[i] = ky_spectrum[i] * dky0
                lnky += dky0
            nky = nky + nky_in
        ky_spectrum = ky_spectrum[:nky]

    elif spectrum_type == 2:
        nky1 = 8
        nky2 = 7
        nky = nky1 + nky2
        ky_spectrum = [0.0] * (nky + nky_in)
        dky_spectrum = [0.0] * (nky + nky_in)

        dky0 = 0.05 * ky_factor / rho_ion
        for i in range(nky1):
            ky_spectrum[i] = (i + 1) * dky0
            dky_spectrum[i] = dky0

        dky0 = 0.2 / rho_ion
        ky0 = ky_spectrum[nky1 - 1]
        for i in range(nky1, nky):
            ky_spectrum[i] = ky0 + (i - nky1 + 1) * dky0
            dky_spectrum[i] = dky0

        ky0 = ky_spectrum[nky - 1] + dky0
        ky1 = 0.4 * ky_factor / rho_e

        if nky_in > 0:
            dky0 = log(ky1 / ky0) / (nky_in - 1)
            lnky = log(ky0)
            for i in range(nky, nky + nky_in):
                ky_spectrum[i] = exp(lnky)
                dky_spectrum[i] = ky_spectrum[i] * dky0
                lnky += dky0
            nky = nky + nky_in
        ky_spectrum = ky_spectrum[:nky]

    elif spectrum_type == 3:
        ky_max_local = ky_factor / rho_ion
        nky1 = int(floor(ky_max_local / ky_in)) - 1
        nky2 = 1
        nky = nky1 + nky2
        ky_spectrum = [0.0] * (nky + nky_in)
        dky_spectrum = [0.0] * (nky + nky_in)

        ky_min_local = ky_in
        dky0 = ky_min_local
        ky_spectrum[0] = ky_min_local
        dky_spectrum[0] = ky_min_local
        for i in range(1, nky1):
            ky_spectrum[i] = ky_spectrum[i - 1] + dky0
            dky_spectrum[i] = dky0

        if ky_spectrum[nky1 - 1] < ky_max_local:
            nky2 = 1
            ky_min_local = ky_spectrum[nky1 - 1]
            dky0 = (ky_max_local - ky_min_local) / nky2
            ky_spectrum[nky1] = ky_spectrum[nky1 - 1] + dky0
            dky_spectrum[nky1] = dky0
        else:
            nky2 = 0
            nky = nky1 + nky2
            ky_max_local = ky_spectrum[nky1 - 1]
            ky_spectrum = ky_spectrum[:nky_in + nky]

        if nky_in > 0:
            ky0 = ky_max_local
            ky1 = 0.4 * ky_factor / rho_e
            lnky = log(ky0 + dky0)
            for i in range(nky, nky + nky_in):
                ky_spectrum[i] = exp(lnky)
                dky_spectrum[i] = ky_spectrum[i] * dky0
                lnky += dky0
            nky = nky + nky_in
        ky_spectrum = ky_spectrum[:nky]

    elif spectrum_type == 4:
        nky1 = 5
        nky2 = 7
        nky = nky1 + nky2

        ky_spectrum = [0.0] * (nky + nky_in)
        dky_spectrum = [0.0] * (nky + nky_in)

        ky_min_local = 0.05 * ky_factor / rho_ion
        for i in range(6):  # i=0..5 -> Julia's 1..6
            ky_spectrum[i] = (i + 1) * ky_min_local
            dky_spectrum[i] = ky_min_local

        ky_min_local = ky_spectrum[5]
        ky_max_local = 1.0 * ky_factor / rho_ion

        dky0 = 0.1 * ky_factor / rho_ion
        # Julia i=7..12 -> Python indices 6..11
        for i in range(nky1 + 1, nky):
            ky_spectrum[i] = ky_min_local + (i - 5) * dky0
            dky_spectrum[i] = dky0

        if nky_in > 0:
            ky0 = 1.0 * ky_factor / rho_ion
            ky1 = 0.4 * ky_factor / rho_e
            dky0 = log(ky1 / ky0) / (nky_in - 1)
            lnky = log(ky0)
            # adjust last linear gap to ky0
            dky_spectrum[nky - 1] = ky0 - ky_spectrum[nky - 1]
            for i in range(nky, nky + nky_in):
                ky_spectrum[i] = exp(lnky)
                dky_spectrum[i] = ky_spectrum[i] * dky0
                lnky += dky0
            nky = nky + nky_in
        ky_spectrum = ky_spectrum[:nky]

    elif spectrum_type == 5:
        raise RuntimeError("spectrum_type == 5 not implemented.")
    else:
        raise ValueError(f"Unknown KYGRID_MODEL: {spectrum_type}")

    return ky_spectrum

# ------- Build the per-sample inputs dict (no dataclasses) -------
def build_inputs_dict(row_31: np.ndarray) -> Dict[str, Any]:
    """
    row_31: shape (31,) for one sample
    returns a dict with required fields for get_ky_spectrum
    """
    TAUS_2 = float(row_31[KEY_TO_COL["TAUS_2"]])
    TAUS_3 = float(row_31[KEY_TO_COL["TAUS_3"]])
    AS_2   = float(row_31[KEY_TO_COL["AS_2"]])
    AS_3   = float(row_31[KEY_TO_COL["AS_3"]])

    TAUS = [GLOBAL_CONST["TAUS_1"], TAUS_2, TAUS_3]
    MASS = [GLOBAL_CONST["MASS_1"], GLOBAL_CONST["MASS_2"], GLOBAL_CONST["MASS_3"]]
    ZS   = [GLOBAL_CONST["ZS_1"],   GLOBAL_CONST["ZS_2"],   GLOBAL_CONST["ZS_3"]]
    AS   = [GLOBAL_CONST["AS_1"],   AS_2,                   AS_3]

    return {
        "UNITS": GLOBAL_CONST["UNITS"],
        "NKY": GLOBAL_CONST["NKY"],
        "KYGRID_MODEL": GLOBAL_CONST["KYGRID_MODEL"],
        "TAUS": TAUS,
        "MASS": MASS,
        "ZS": ZS,
        "AS": AS,
        "NS": GLOBAL_CONST["NS"],
        "USE_AVE_ION_GRID": GLOBAL_CONST["USE_AVE_ION_GRID"],
    }

def compute_ky_matrix_skip_bad(data_2d: np.ndarray, grad_r0: float) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Try each sample; skip any that raise due to domain errors or invalid params.
    Returns:
      ky_mat:    (kept_samples, nky_total)
      inputs_kept: (kept_samples, 31)
      kept_idx:  list of original row indices kept
      skipped_idx: list of original row indices skipped
    """
    data_2d = np.asarray(data_2d, dtype=float)
    assert data_2d.ndim == 2 and data_2d.shape[1] == 31, "Input must be (samples, 31)."

    kept_idx: List[int] = []
    skipped_idx: List[int] = []
    ky_rows: List[np.ndarray] = []
    inputs_rows: List[np.ndarray] = []
    nky_total: int = -1

    for i in range(data_2d.shape[0]):
        row = data_2d[i]
        try:
            # quick sanity: TAUS_2 and TAUS_3 must be positive
            if row[KEY_TO_COL["TAUS_2"]] <= 0 or row[KEY_TO_COL["TAUS_3"]] <= 0:
                raise ValueError("Non-positive TAUS_2/TAUS_3")

            inputs_i = build_inputs_dict(row)
            ky_i = get_ky_spectrum(inputs_i, grad_r0)

            if nky_total < 0:
                nky_total = len(ky_i)
            elif len(ky_i) != nky_total:
                raise ValueError(f"Inconsistent ky length ({len(ky_i)} != {nky_total})")

            ky_rows.append(np.array(ky_i, dtype=float))
            inputs_rows.append(row.copy())
            kept_idx.append(i)

        except Exception as e:
            print(f"Skipping sample {i}: {e}")
            skipped_idx.append(i)
            continue

    if len(kept_idx) == 0:
        raise RuntimeError("No valid samples after skipping problematic rows.")

    ky_mat = np.vstack(ky_rows)                     # (kept, nky_total)
    inputs_kept = np.vstack(inputs_rows)            # (kept, 31)
    return ky_mat, inputs_kept, kept_idx, skipped_idx

def load_npy_or_npz(path: str) -> np.ndarray:
    """
    Loads .npy or .npz (takes the first array from .npz).
    Ensures output is 2D (samples, 31).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".npz":
        z = np.load(path, allow_pickle=False)
        key = list(z.keys())[0]
        arr = z[key]
    else:
        raise ValueError("Only .npy and .npz are supported here.")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size != 31:
            raise ValueError(f"1D array must have size 31; got {arr.size}.")
        arr = arr.reshape(1, 31)
    if arr.shape[1] != 31:
        raise ValueError(f"Expected shape (samples, 31); got {arr.shape}.")
    return arr

def save_h5(out_path: str, inputs_mat: np.ndarray, ky_mat: np.ndarray, grad_r0: float,
            kept_idx: List[int], skipped_idx: List[int]):
    """
    Writes an HDF5 with:
      - root datasets named by INPUT_KEYS, each shape (kept_samples,)
      - dataset 'ky' with shape (kept_samples, nky_total)
      - datasets 'kept_idx' and 'skipped_idx' to map back to original rows
    """
    samples, cols = inputs_mat.shape
    assert cols == len(INPUT_KEYS), "inputs_mat second axis must match INPUT_KEYS length."

    with h5py.File(out_path, "w") as f:
        # Inputs (one dataset per key)
        for j, name in enumerate(INPUT_KEYS):
            f.create_dataset(name, data=inputs_mat[:, j], dtype='f8')

        # ky matrix
        f.create_dataset("ky", data=ky_mat, dtype='f8')

        # index mapping
        f.create_dataset("kept_idx", data=np.array(kept_idx, dtype=np.int64))
        f.create_dataset("skipped_idx", data=np.array(skipped_idx, dtype=np.int64))

        # optional metadata as file attributes
        f.attrs["grad_r0"] = float(grad_r0)
        f.attrs["UNITS"] = GLOBAL_CONST["UNITS"]
        f.attrs["NKY"] = GLOBAL_CONST["NKY"]
        f.attrs["KYGRID_MODEL"] = GLOBAL_CONST["KYGRID_MODEL"]
        f.attrs["USE_AVE_ION_GRID"] = int(GLOBAL_CONST["USE_AVE_ION_GRID"])
        f.attrs["NS"] = GLOBAL_CONST["NS"]

if __name__ == "__main__":
    npy_file = "input_generation/samples_10k_minmax_normal.npy"
    grad_r0 = 1.23314445670738

    data = load_npy_or_npz(npy_file)
    ky_mat, inputs_kept, kept_idx, skipped_idx = compute_ky_matrix_skip_bad(data, grad_r0)

    print(f"Computed ky for {ky_mat.shape[0]} samples, skipped {len(skipped_idx)}.")
    print(f"ky shape: {ky_mat.shape}")

    out_h5 = "./out_10k_minmax_norm.h5"
    save_h5(out_h5, inputs_kept, ky_mat, grad_r0, kept_idx, skipped_idx)
    print(f"Wrote HDF5: {out_h5}")
