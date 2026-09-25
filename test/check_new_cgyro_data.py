"""Verify a CGYRO `rho/` drop before it is used for anything.

Run this on every new data drop. The reason is 0.1: TGLF and CGYRO store `sumf`
with the species and field axes transposed, both of CGYRO's middle axes are 3,
and so reading it with the wrong layout **raises no shape error** -- it silently
sums species and indexes fields as if they were species. That mis-derivation
survived months and produced the retracted "Pi R^2 ~ 0 is a modelling wall"
finding. A run exiting 0 proves nothing about whether its targets were right.

Checks, in the order they can bite:

1. **Layout.** Derive the four channels under BOTH the CGYRO layout (sum the
   field axis, what `src/dataset/CGYRO_Spectra.py` does) and the TGLF layout
   (sum the species axis), and correlate each against the file's own `OUT_*`
   totals. CGYRO layout must give corr 1.000000 on all four. The TGLF-layout
   column is printed alongside deliberately: seeing Pi collapse to ~0.6 while
   Ge/Qe stay ~0.99 is the signature to recognise, and its absence would mean
   the file is not in the layout this repo expects at all.
2. **Row/group counts**, and how many rows are genuinely new physics against the
   previous drop -- identity is SHA1 of the 31 input columns as raw float32
   (`src/utils/Tracker.py`), exact bit match, no tolerance.
3. **Duplicate physics hashes**, including duplicate inputs carrying DIFFERENT
   fluxes, which exist in this dataset and which any leak-free split must group.
4. **`failed_mask` rate** and rho-band coverage, both of which drive the split
   maker's fold balancing (it balances on VALID-KY count, not rows).

Nothing here writes to the dataset or to S3; it only reads and reports.
"""
import os
import glob
import hashlib

import numpy as np
import h5py

# === USER INPUT ===
NEW_DIR = "/data/lucas_work/cgyro-rho-sep-new/rho"
OLD_DIR = "/data/lucas_work/cgyro-rho-sep/rho"
OUT_KEYS = ("OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i")
CHANNELS = ("Ge", "Qe", "Qi", "Pi")
SUMF_KEY = "sumf"
MASK_KEY = "meta/failed_mask"
RHO_KEY = "rho"
# `sumf` is (N, nky, 2, ns=3, nf=3, 5). The repo slices [:, :, 0, ...] first
# (index 1 of that axis is zero padding), leaving (N, nky, ns, nf, 5); both axis
# constants below index that POST-SLICE array, exactly as
# src/dataset/CGYRO_Spectra.py:58-62 does. Do not subtract 1 for the dropped
# axis -- that was a bug here on 2026-09-25 and it made both layouts sum species,
# producing two identical columns and a spurious FAIL.
SLICE_AXIS = 2      # leading axis size 2 in CGYRO; index 1 is zero padding
FIELD_AXIS = 3      # post-slice: nf  (CGYRO layout -- correct for this data)
SPECIES_AXIS = 2    # post-slice: ns  (TGLF layout -- the wrong-layout signature)
INPUT_KEYS = [
    "AS_2", "AS_3", "BETAE_log10", "DEBYE_log10", "DELTA_LOC", "DRMAJDX_LOC",
    "DZMAJDX_LOC", "KAPPA_LOC", "P_PRIME_LOC", "Q_LOC", "Q_PRIME_LOC",
    "RLNS_1", "RLNS_2", "RLNS_3", "RLTS_1", "RLTS_2", "RLTS_3", "RMAJ_LOC",
    "RMIN_LOC", "S_DELTA_LOC", "S_KAPPA_LOC", "S_ZETA_LOC", "TAUS_2", "TAUS_3",
    "VEXB_SHEAR", "VPAR_1", "VPAR_SHEAR_1", "XNUE_log10", "ZEFF", "ZETA_LOC",
    "ZMAJ_LOC",
]
# ==================


def derive(sumf, field_axis):
    """(N, nky, 4) channels [Ge, Qe, Qi, Pi] summing `field_axis` of the slice."""
    sl = np.take(sumf, 0, axis=SLICE_AXIS)          # (N, nky, ns, nf, 5)
    flux = np.sum(sl, axis=field_axis)              # axis indexes the SLICED array
    # species 0 = electrons, 1: = ions; moment 0 = particle, 1 = heat, 2 = momentum
    return np.stack([flux[:, :, 0, 0], flux[:, :, 0, 1],
                     flux[:, :, 1:, 1].sum(-1), flux[:, :, 1:, 2].sum(-1)], axis=-1)


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def hashes(f):
    """SHA1 of the 31 inputs as raw float32 -- the project's physics identity."""
    cols = [np.asarray(f[k], dtype=np.float32) for k in INPUT_KEYS if k in f]
    if len(cols) != len(INPUT_KEYS):
        missing = [k for k in INPUT_KEYS if k not in f]
        raise KeyError(f"missing input keys: {missing[:5]}")
    X = np.stack(cols, axis=1)
    return [hashlib.sha1(r.tobytes()).hexdigest() for r in X], X


def scan(d):
    files = sorted(glob.glob(os.path.join(d, "*.h5")))
    rows = 0
    H, RHO = [], []
    agg = {c: {"cg": [], "tg": [], "out": []} for c in CHANNELS}
    mask_tot = mask_bad = 0
    bad_files = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                sumf = np.asarray(f[SUMF_KEY])
                h, _ = hashes(f)
                H += h
                rows += sumf.shape[0]
                if RHO_KEY in f:
                    RHO += list(np.asarray(f[RHO_KEY]).ravel())
                if MASK_KEY in f:
                    m = np.asarray(f[MASK_KEY])
                    mask_tot += m.size
                    mask_bad += int(np.count_nonzero(m))
                cg = derive(sumf, FIELD_AXIS).sum(axis=1)     # sum over ky -> totals
                tg = derive(sumf, SPECIES_AXIS).sum(axis=1)
                for i, c in enumerate(CHANNELS):
                    if OUT_KEYS[i] in f:
                        agg[c]["out"].append(np.asarray(f[OUT_KEYS[i]]).ravel())
                        agg[c]["cg"].append(cg[:, i])
                        agg[c]["tg"].append(tg[:, i])
        except Exception as e:
            bad_files.append((os.path.basename(p), repr(e)[:90]))
    return dict(files=files, rows=rows, H=H, RHO=RHO, agg=agg,
                mask_tot=mask_tot, mask_bad=mask_bad, bad=bad_files)


def main():
    print(f"NEW: {NEW_DIR}\nOLD: {OLD_DIR}\n")
    new = scan(NEW_DIR)
    old = scan(OLD_DIR) if os.path.isdir(OLD_DIR) else None

    print("=" * 74)
    print("1. LAYOUT — derived channels vs the files' own OUT_* totals")
    print("   CGYRO layout must be 1.000000 on all four. The TGLF column is the")
    print("   wrong-layout signature, shown so it can be recognised, not used.")
    print(f"   {'channel':<9}{'CGYRO layout':>15}{'TGLF layout':>15}{'n rows':>10}")
    ok = True
    for c in CHANNELS:
        a = new["agg"][c]
        if not a["out"]:
            print(f"   {c:<9}{'NO OUT_ KEY':>15}")
            ok = False
            continue
        out = np.concatenate(a["out"])
        ccg, ctg = corr(np.concatenate(a["cg"]), out), corr(np.concatenate(a["tg"]), out)
        flag = "" if ccg > 0.9999 else "   <-- FAIL"
        ok &= ccg > 0.9999
        print(f"   {c:<9}{ccg:>15.6f}{ctg:>15.6f}{out.size:>10}{flag}")
    print(f"   verdict: {'PASS — CGYRO layout confirmed' if ok else 'FAIL — DO NOT TRAIN ON THIS'}")

    print("\n" + "=" * 74)
    print("2. SIZE")
    u_new = len(set(new["H"]))
    print(f"   files {len(new['files']):>6}   rows {new['rows']:>7}   unique physics hashes {u_new:>7}")
    if old:
        u_old = len(set(old["H"]))
        gained = len(set(new["H"]) - set(old["H"]))
        lost = len(set(old["H"]) - set(new["H"]))
        print(f"   old:  files {len(old['files']):>6}   rows {old['rows']:>7}   unique {u_old:>7}")
        print(f"   delta: +{len(new['files']) - len(old['files'])} files, "
              f"+{new['rows'] - old['rows']} rows, +{u_new - u_old} unique")
        print(f"   genuinely NEW physics hashes: {gained}"
              + (f"   |   hashes PRESENT BEFORE AND NOW GONE: {lost}  <-- investigate"
                 if lost else "   |   no hashes lost (old set is a subset)"))

    print("\n" + "=" * 74)
    print("3. DUPLICATE PHYSICS (these must be grouped by any leak-free split)")
    _, counts = np.unique(new["H"], return_counts=True)
    dups = int((counts > 1).sum())
    print(f"   {dups} hashes appear more than once; {int(counts[counts > 1].sum())} rows involved")
    print(f"   max copies of one hash: {int(counts.max())}")

    print("\n" + "=" * 74)
    print("4. MASK AND RHO")
    if new["mask_tot"]:
        print(f"   failed_mask: {new['mask_bad']}/{new['mask_tot']} ky entries "
              f"= {100.0 * new['mask_bad'] / new['mask_tot']:.1f}% non-converged")
    if new["RHO"]:
        v, c = np.unique(np.round(np.array(new["RHO"], float), 2), return_counts=True)
        print("   rho bands: " + "  ".join(f"{a}:{b}" for a, b in zip(v, c)))
        print(f"   bands present: {len(v)}  (the split maker needs all of them in both halves)")
    if new["bad"]:
        print(f"\n   UNREADABLE FILES ({len(new['bad'])}):")
        for n, e in new["bad"]:
            print(f"     {n}  {e}")


if __name__ == "__main__":
    main()
