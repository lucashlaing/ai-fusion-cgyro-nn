#!/usr/bin/env python3
# make_gaussian_samples_by_rho.py
import json, os
from pathlib import Path
import numpy as np
from typing import Dict, List

# Keep your fixed column order
INPUT_KEYS: List[str] = [
    "RLTS_3","KAPPA_LOC","ZETA_LOC","TAUS_3","VPAR_1","Q_LOC","RLNS_1","TAUS_2",
    "Q_PRIME_LOC","P_PRIME_LOC","ZMAJ_LOC","VPAR_SHEAR_1","RLTS_2","S_DELTA_LOC",
    "RLTS_1","RMIN_LOC","DRMAJDX_LOC","AS_3","RLNS_3","DZMAJDX_LOC","DELTA_LOC",
    "S_KAPPA_LOC","ZEFF","VEXB_SHEAR","RMAJ_LOC","AS_2","RLNS_2","S_ZETA_LOC",
    "BETAE_log10","XNUE_log10","DEBYE_log10"
]

def truncated_normal(mean: float, std: float, low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Normal(mean, std) truncated to [low, high] via rejection sampling."""
    if std <= 0:
        return np.full(size, np.clip(mean, low, high), dtype=float)

    out = np.empty(size, dtype=float)
    remaining = np.ones(size, dtype=bool)
    attempts = 0
    while remaining.any():
        n = int(remaining.sum())
        cand = rng.normal(loc=mean, scale=std, size=max(n, 1024))
        cand = cand[(cand >= low) & (cand <= high)]
        take = min(cand.size, n)
        if take > 0:
            idx = np.where(remaining)[0][:take]
            out[idx] = cand[:take]
            remaining[idx] = False
        attempts += 1
        if attempts > 100:
            # fallback: clip a fresh normal draw
            draw = rng.normal(loc=mean, scale=std, size=n)
            out[remaining] = np.clip(draw, low, high)
            remaining[:] = False
    return out

def _validate_rho_block(rho_label: str, block: Dict):
    for k in INPUT_KEYS:
        if k not in block:
            raise KeyError(f"[rho={rho_label}] Key '{k}' missing from JSON.")
        for fld in ("min", "max", "mean", "std"):
            if fld not in block[k]:
                raise KeyError(f"[rho={rho_label}] Key '{k}' missing field '{fld}'.")

def generate_samples(json_path: str, out_dir: str, n: int = 10000, seed: int = 42):
    """
    Read per-rho stats JSON and write one .npy per rho into out_dir.
    Each array has shape (n, len(INPUT_KEYS)) in the order of INPUT_KEYS.
    """
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r") as f:
        # Expected structure: { "0.2": { PARAM: {min,max,mean,std,...}, ... }, "0.3": {...}, ... }
        stats_by_rho: Dict[str, Dict] = json.load(f)

    # Validate and sort rho labels numerically where possible
    def _rho_sort_key(s: str):
        try:
            return float(s)
        except Exception:
            return float("inf")

    rho_labels = sorted(stats_by_rho.keys(), key=_rho_sort_key)

    rng = np.random.default_rng(seed)

    # Write a column order manifest once
    (out_dir / "COLUMN_ORDER.json").write_text(json.dumps(INPUT_KEYS, indent=2))

    for rho_label in rho_labels:
        block = stats_by_rho[rho_label]
        _validate_rho_block(rho_label, block)

        cols = []
        for k in INPUT_KEYS:
            s = block[k]
            lo, hi = float(s["min"]), float(s["max"])
            mu, sd = float(s["mean"]), float(s["std"])
            if hi < lo:
                lo, hi = hi, lo
            if hi == lo:
                samples = np.full(n, lo, dtype=float)
            else:
                samples = truncated_normal(mu, max(sd, 1e-12), lo, hi, n, rng)
            cols.append(samples)

        arr = np.stack(cols, axis=1)

        # Build a readable filename, e.g., samples_rho_0.3.npy (preserve label as-is)
        safe_rho = str(rho_label).replace("/", "_")
        out_file = out_dir / f"samples_rho_{safe_rho}.npy"
        np.save(out_file, arr)

        # Optional: sidecar metadata for traceability
        meta = {
            "rho_label": rho_label,
            "n_rows": int(arr.shape[0]),
            "n_cols": int(arr.shape[1]),
            "columns": INPUT_KEYS,
            "source_json": str(json_path)
        }
        (out_dir / f"samples_rho_{safe_rho}.meta.json").write_text(json.dumps(meta, indent=2))

        print(f"✅ [rho={rho_label}] wrote {arr.shape} → {out_file}")

if __name__ == "__main__":
    # === Parameters here ===
    JSON_FILE = "/Users/wesleyliu/Documents/Github/gacode-docker/input_generation/mean_std_with_rho (1).json"
    OUT_DIR   = "/Users/wesleyliu/Documents/Github/gacode-docker/input_generation/samples_by_rho"
    N_SAMPLES = 10000
    SEED      = 123

    main(JSON_FILE, OUT_DIR, N_SAMPLES, SEED)
