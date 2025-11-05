#!/usr/bin/env python3
# make_gaussian_samples.py
import json, os
import numpy as np

INPUT_KEYS = [
    "RLTS_3","KAPPA_LOC","ZETA_LOC","TAUS_3","VPAR_1","Q_LOC","RLNS_1","TAUS_2",
    "Q_PRIME_LOC","P_PRIME_LOC","ZMAJ_LOC","VPAR_SHEAR_1","RLTS_2","S_DELTA_LOC",
    "RLTS_1","RMIN_LOC","DRMAJDX_LOC","AS_3","RLNS_3","DZMAJDX_LOC","DELTA_LOC",
    "S_KAPPA_LOC","ZEFF","VEXB_SHEAR","RMAJ_LOC","AS_2","RLNS_2","S_ZETA_LOC",
    "BETAE_log10","XNUE_log10","DEBYE_log10"
]

def truncated_normal(mean, std, low, high, size, rng):
    """Sample Normal(mean, std) truncated to [low, high] with rejection sampling."""
    if std <= 0:
        return np.full(size, np.clip(mean, low, high), dtype=float)

    out = np.empty(size, dtype=float)
    remaining = np.ones(size, dtype=bool)
    attempts = 0
    while remaining.any():
        n = remaining.sum()
        cand = rng.normal(loc=mean, scale=std, size=max(n, 1024))
        cand = cand[(cand >= low) & (cand <= high)]
        take = min(cand.size, n)
        if take > 0:
            out[np.where(remaining)[0][:take]] = cand[:take]
            remaining[np.where(remaining)[0][:take]] = False
        attempts += 1
        if attempts > 100:
            out[remaining] = np.clip(
                rng.normal(loc=mean, scale=std, size=remaining.sum()), low, high
            )
            remaining[:] = False
    return out

def main(json_path: str, out_path: str, n: int = 10000, seed: int = 42):
    with open(json_path, "r") as f:
        stats = json.load(f)

    # Validate JSON has everything
    for k in INPUT_KEYS:
        if k not in stats:
            raise KeyError(f"Key '{k}' missing from JSON.")
        for fld in ("min","max","mean","std"):
            if fld not in stats[k]:
                raise KeyError(f"Key '{k}' missing field '{fld}'.")

    rng = np.random.default_rng(seed)
    cols = []
    for k in INPUT_KEYS:
        s = stats[k]
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

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, arr)
    print(f"✅ Wrote array {arr.shape} to {out_path}")
    print("Column order:", INPUT_KEYS)

if __name__ == "__main__":
    # === Parameters here ===
    JSON_FILE = "/Users/wesleyliu/Documents/Github/gacode-docker/input_generation/merged_dist.json"
    OUT_FILE  = "/Users/wesleyliu/Documents/Github/gacode-docker/input_generation/samples_10k_minmax_normal.npy"
    N_SAMPLES = 1000
    SEED      = 123

    main(JSON_FILE, OUT_FILE, N_SAMPLES, SEED)
