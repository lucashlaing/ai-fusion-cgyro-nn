#!/usr/bin/env python
"""
Build trainable coreset splits from the TGLF pool -- SELF-CONTAINED.

The selections are COMPUTED here from a fixed seed (nothing is read from a
precomputed .npz), so this runs unchanged inside a Nautilus pod.

Arms
----
random        uniform without replacement, weight 1.0
lightweight   Bachem/Lucic/Krause (KDD 2018) lightweight coreset:
                  q_i = 0.5/n + 0.5*d2_i/sum(d2),  d2_i = ||z_i - mean(Z)||^2
                  w   = 1/(K*q[sel])
leverage_rff  Drineas et al. (SODA 2006) leverage sampling on a random-Fourier
              lift: P = cos(Z@W + b) with Dr=512, A = P^T P,
                  lev_i = p_i^T A^{-1} p_i,  p = lev/sum(lev),  w = 1/(K*p[sel])

Z is the z-scored matrix of the 31 scalar inputs over the FULL pool.

Output (drop-in `dataset_root` for `src/offline_train.py`, which reads the
"pool" split and trains from scratch):

    <OUT_ROOT>/<arm>/pool/coreset.h5     selected rows + sample_weight
    <OUT_ROOT>/<arm>/test/*.h5           symlinks to the 9 real test files

Config is hardcoded below on purpose (house rule: one-off scripts do NOT add
run_config yaml keys). The data location is overridable by env var so the same
file works locally and on the pod, where the pool lands at ./data/pool.

Run:
    /data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/bin/python test/make_coreset_split.py
"""

import glob
import os
import sys
import time

import h5py
import numpy as np

# --------------------------------------------------------------------------
# CONFIG (hardcoded; env vars only relocate the data)
# --------------------------------------------------------------------------
DATA_ROOT = os.environ.get(
    "CORESET_DATA_ROOT",
    "/data/lucas_work/tglf_sumf_data_major_minor_perturb_madcut_filter",
)
# On the Nautilus pod: CORESET_DATA_ROOT=./data  (pool/ + test/ live there).
POOL_DIR = os.environ.get("CORESET_POOL_DIR", os.path.join(DATA_ROOT, "pool"))
TEST_DIR = os.environ.get("CORESET_TEST_DIR", os.path.join(DATA_ROOT, "test"))
OUT_ROOT = os.environ.get("CORESET_OUT_ROOT", "/data/lucas_work/tglf_coreset_splits")

SEED = 0
K = 320000          # coreset size, per arm
RFF_DIM = 512       # Dr
LIFT_CHUNK = 400000  # pool rows per chunk of the random-Fourier lift
ARMS = ["random", "lightweight", "leverage_rff"]

# Row counts of sorted(glob(POOL_DIR/*.h5)) in that exact order. The selection
# indices address the concatenation in this order, so a mismatch makes every
# index meaningless -> hard abort.
EXPECTED_COUNTS = [443968, 393280, 432914, 411714, 447578, 359130, 424856, 404591, 417258]

# The 31 scalar inputs, verbatim from run_configs/dataset/SiNN_local.yaml.
INPUT_KEYS = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1",
    "RLTS_2", "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3",
    "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR",
    "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10",
    "DEBYE_log10",
]
# Copied alongside the inputs. ql/gw/pots are deliberately NOT copied (unused by
# the datapipe, and huge). No `meta` group is written -- the TGLF pool has none
# and the datapipe defaults the failed_mask to all-valid.
# OUT_* are the precomputed ky-summed totals (verified identical to the derived
# channels, CLAUDE.md). They are (n,) float64 -- 2.6 MB each -- and keeping them
# means downstream analysis reads ~10 MB instead of the ~1 GB sumf per arm.
# ky and sumf are MANDATORY (the datapipe derives inputs/targets from them).
# rho and OUT_* are OPTIONAL and resolved against the actual files at runtime:
# the S3 copy of this dataset PREDATES test/add_rho_key.py, so on a Nautilus pod
# data/pool/*.h5 have NO `rho` -- while the local copy does. The datapipe never
# reads rho (only input_keys + ky + sumf + meta/mask), so its absence is
# harmless; hard-requiring it is what killed the first pod run.
REQUIRED_ARRAY_KEYS = ["ky", "sumf"]
OPTIONAL_ARRAY_KEYS = ["rho", "OUT_G_elec", "OUT_Q_elec", "OUT_Q_ions", "OUT_P_ions"]
ARRAY_KEYS = REQUIRED_ARRAY_KEYS + OPTIONAL_ARRAY_KEYS   # pruned by resolve_array_keys()

DIAG_KEY = "OUT_Q_ions"   # end-to-end sanity diagnostic (top-1% |Qi| enrichment)
# Source rows per sequential slab read (~115 MB of sumf). The copy is done as
# ONE sequential pass over the pool that fills every arm at once: scattered
# per-row h5 fancy indexing over NFS runs ~100x slower than streaming the file.
SLAB = 20000

# --------------------------------------------------------------------------


def resolve_array_keys(files):
    """Keep only the optional arrays this copy of the dataset actually has."""
    global ARRAY_KEYS, DIAG_KEY
    with h5py.File(files[0], "r") as f:
        present = set(f.keys())
    missing_required = [k for k in REQUIRED_ARRAY_KEYS if k not in present]
    if missing_required:
        raise SystemExit(f"ABORT: pool files lack mandatory dataset(s) {missing_required}")
    have = [k for k in OPTIONAL_ARRAY_KEYS if k in present]
    lost = [k for k in OPTIONAL_ARRAY_KEYS if k not in present]
    ARRAY_KEYS = REQUIRED_ARRAY_KEYS + have
    if lost:
        print(f"note: optional dataset(s) {lost} absent from this copy -- not copied "
              f"(the datapipe does not read them)")
    if DIAG_KEY not in present:
        print(f"note: {DIAG_KEY} absent -- enrichment diagnostic disabled")
        DIAG_KEY = None
    return ARRAY_KEYS


def pool_files():
    files = sorted(glob.glob(os.path.join(POOL_DIR, "*.h5")))
    if len(files) != len(EXPECTED_COUNTS):
        raise SystemExit(
            f"ABORT: expected {len(EXPECTED_COUNTS)} pool files in {POOL_DIR}, found {len(files)}"
        )
    return files


def verify_counts(files):
    counts = []
    for p in files:
        with h5py.File(p, "r") as f:
            counts.append(int(f["ky"].shape[0]))
    if counts != EXPECTED_COUNTS:
        raise SystemExit(
            "ABORT: pool row counts / file order do not match the expected list.\n"
            f"  got      {counts}\n  expected {EXPECTED_COUNTS}\n"
            f"  files    {[os.path.basename(x) for x in files]}"
        )
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    print(f"verified {len(files)} pool files in {POOL_DIR}, total rows = {offsets[-1]}")
    return offsets


def load_inputs(files, n_total):
    """Z-scored (n, 31) float64 matrix of the 31 scalar inputs, plus the diagnostic."""
    t0 = time.time()
    X = np.empty((n_total, len(INPUT_KEYS)), dtype=np.float64)
    diag = np.empty(n_total, dtype=np.float64)
    pos = 0
    for p in files:
        with h5py.File(p, "r") as f:
            n = f["ky"].shape[0]
            for j, k in enumerate(INPUT_KEYS):
                X[pos:pos + n, j] = np.asarray(f[k], dtype=np.float64)
            if DIAG_KEY is not None:
                diag[pos:pos + n] = np.asarray(f[DIAG_KEY], dtype=np.float64)
            pos += n
    assert pos == n_total
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    const = sd <= 0
    if const.any():
        names = [INPUT_KEYS[j] for j in np.nonzero(const)[0]]
        print(f"  note: constant input column(s) {names} -- scale forced to 1.0")
    sd = np.where(const, 1.0, sd)
    Z = (X - mu) / sd
    del X
    print(f"loaded + standardized inputs: {Z.shape} in {time.time() - t0:.1f}s")
    return Z, diag



def draws_for_unique(prob, target, n):
    """
    Number of i.i.d. WITH-REPLACEMENT draws whose expected unique-row count is
    `target`.  E[unique] = n - sum_i (1-p_i)^m, monotone in m, so bisect.

    Drawing m > target and collapsing duplicates leaves sum(w) = n unchanged
    (w = 1/(m*p) shrinks exactly as the draw count grows), so the coreset's
    total mass -- and its guarantee -- are identical.  Only the number of
    STORED rows rises, which is what makes the arms budget-comparable: every
    sample count in this repo is a unique count.
    """
    lo, hi = target, 20 * target
    log1m = np.log1p(-prob)
    for _ in range(60):
        mid = (lo + hi) // 2
        exp_u = n - np.exp(mid * log1m).sum()
        if exp_u < target:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)

def sel_random(n, rng):
    sel = rng.choice(n, size=K, replace=False)
    return sel.astype(np.int64), np.ones(K, dtype=np.float64)


def sel_lightweight(Z, rng):
    """Bachem/Lucic/Krause KDD 2018 lightweight coreset."""
    n = Z.shape[0]
    mu = Z.mean(axis=0)
    d2 = np.einsum("ij,ij->i", Z - mu, Z - mu)
    q = 0.5 / n + 0.5 * d2 / d2.sum()
    q = q / q.sum()
    # WITH replacement: w = 1/(K*q) is the i.i.d.-with-replacement (Horvitz-
    # Thompson) estimator, and it is only unbiased for that design. Drawing
    # without replacement renormalizes q after every draw, which lifts the
    # inclusion probability of low-q rows above K*q_i and makes 1/(K*q_i)
    # over-count them -- measured as sum(w)/n = 1.24 on the leverage arm.
    # Both Bachem et al. (KDD 2018) and Drineas et al. (SODA 2006) specify
    # i.i.d. with replacement. Duplicate draws are correct, not a defect:
    # they mean that row is seen proportionally more often during training.
    m = draws_for_unique(q, K, n)
    print(f"  lightweight: {m} draws -> ~{K} unique rows")
    sel = rng.choice(n, size=m, replace=True, p=q)
    w = 1.0 / (m * q[sel])
    # The 1/2-uniform term floors q_i at 0.5/n, so no weight can exceed 2n/K.
    cap = 2.0 * n / m
    assert w.max() <= cap * (1 + 1e-9), f"lightweight weight {w.max()} exceeds cap 2n/m={cap}"
    print(f"  lightweight: max(w)={w.max():.6f} <= 2n/m={cap:.6f}  OK")
    _report_mass(n, sel, w, "lightweight")
    return sel.astype(np.int64), w


def _report_mass(n, sel, w, name):
    """The defining coreset property: the weights must reconstruct the full
    pool's total mass, i.e. sum(w) ~= n. Anything far from 1.0 means the
    sampling design and the weight formula disagree."""
    r = w.sum() / n
    print(f"  {name}: sum(w)={w.sum():.6g}  sum(w)/n={r:.4f}  "
          f"unique rows={len(np.unique(sel))} of {len(sel)}")
    assert 0.95 < r < 1.05, f"{name}: sum(w)/n = {r:.4f}, expected ~1.0"


def sel_leverage_rff(Z, rng):
    """Drineas et al. SODA 2006 leverage sampling on a random-Fourier lift."""
    n, d = Z.shape
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d), size=(d, RFF_DIM))
    b = rng.uniform(0.0, 2.0 * np.pi, size=RFF_DIM)

    # pass 1: A = sum_chunks P^T P   (float64)
    t0 = time.time()
    A = np.zeros((RFF_DIM, RFF_DIM), dtype=np.float64)
    for s in range(0, n, LIFT_CHUNK):
        P = np.cos(Z[s:s + LIFT_CHUNK] @ W + b)
        A += P.T @ P
    print(f"  leverage_rff: A accumulated in {time.time() - t0:.1f}s, cond={np.linalg.cond(A):.3e}")

    Ainv = np.linalg.inv(A)
    Ainv = 0.5 * (Ainv + Ainv.T)  # symmetrize

    # pass 2: lev_i = p_i^T A^{-1} p_i
    lev = np.empty(n, dtype=np.float64)
    for s in range(0, n, LIFT_CHUNK):
        P = np.cos(Z[s:s + LIFT_CHUNK] @ W + b)
        lev[s:s + P.shape[0]] = np.einsum("ij,ij->i", P @ Ainv, P)

    # sum of leverage scores == rank == RFF_DIM
    tot = lev.sum()
    # sum of leverage scores == RANK of the lifted matrix == min(n, RFF_DIM).
    # On the real pool n >> RFF_DIM so this is exactly RFF_DIM; the min() only
    # matters for degenerate/toy inputs, where a rank-deficient lift is correct
    # behaviour rather than an error.
    expected_rank = min(n, RFF_DIM)
    if n < 10 * RFF_DIM:
        # Degenerate lift (toy inputs): A is rank-deficient, its inverse is
        # numerically meaningless, and sum(lev) is noise. Warn, do not assert.
        print(f"  leverage_rff: WARNING rank-deficient lift (n={n} < 10*{RFF_DIM}); "
              f"sum(lev)={tot:.6f} vs rank {expected_rank} -- check skipped")
    else:
        assert abs(tot - expected_rank) < 1e-3 * expected_rank, \
            f"sum(lev)={tot!r}, expected rank {expected_rank}"
    print(f"  leverage_rff: sum(lev)={tot:.9f} (rank {expected_rank}); "
          f"lev in [{lev.min():.3e}, {lev.max():.3e}]")
    assert lev.min() > 0

    p = lev / tot
    m = draws_for_unique(p, K, n)
    print(f"  leverage_rff: {m} draws -> ~{K} unique rows")
    sel = rng.choice(n, size=m, replace=True, p=p)
    w = 1.0 / (m * p[sel])
    _report_mass(n, sel, w, "leverage_rff")
    return sel.astype(np.int64), w


def make_test_symlinks(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(TEST_DIR, "*.h5")))
    if not srcs:
        raise SystemExit(f"ABORT: no test h5 files in {TEST_DIR}")
    for s in srcs:
        dst = os.path.join(out_dir, os.path.basename(s))
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(s), dst)
    return len(srcs)


def write_all_arms(selections, files, offsets):
    """
    Copy the selected rows verbatim into <OUT_ROOT>/<arm>/pool/coreset.h5 for
    every arm, in ONE sequential pass over the pool files.

    selections: dict arm -> (sel, w), sel ascending global row indices.
    Returns dict arm -> (out_path, nbytes, nlinks).
    """
    all_keys = INPUT_KEYS + ARRAY_KEYS
    t0 = time.time()
    handles, dsets, plan, paths = {}, {}, {}, {}

    with h5py.File(files[0], "r") as f0:
        shapes = {k: f0[k].shape[1:] for k in all_keys}
        dtypes = {k: f0[k].dtype for k in all_keys}

    try:
        for arm, (sel, w) in selections.items():
            pool_out = os.path.join(OUT_ROOT, arm, "pool")
            os.makedirs(pool_out, exist_ok=True)
            paths[arm] = os.path.join(pool_out, "coreset.h5")
            fo = h5py.File(paths[arm], "w")
            handles[arm] = fo
            dsets[arm] = {
                k: fo.create_dataset(k, shape=(len(sel),) + shapes[k], dtype=dtypes[k])
                for k in all_keys
            }
            fo.create_dataset("sample_weight", data=w.astype(np.float64), dtype=np.float64)
            # per source file: (local row indices ascending, first output row)
            file_of = np.searchsorted(offsets, sel, side="right") - 1
            plan[arm] = {}
            for fi in range(len(files)):
                m = file_of == fi
                if not m.any():
                    continue
                out_pos = np.nonzero(m)[0]            # contiguous ascending
                local = (sel[m] - offsets[fi]).astype(np.int64)
                assert np.all(np.diff(local) > 0)
                plan[arm][fi] = (local, int(out_pos[0]))

        for fi, path in enumerate(files):
            n_src = int(offsets[fi + 1] - offsets[fi])
            with h5py.File(path, "r") as fsrc:
                for s in range(0, n_src, SLAB):
                    e = min(s + SLAB, n_src)
                    # which arms want anything from this slab
                    todo = []
                    for arm in selections:
                        if fi not in plan[arm]:
                            continue
                        local, out0 = plan[arm][fi]
                        a = int(np.searchsorted(local, s, side="left"))
                        b = int(np.searchsorted(local, e, side="left"))
                        if b > a:
                            todo.append((arm, local[a:b] - s, out0 + a, out0 + b))
                    if not todo:
                        continue
                    for k in all_keys:
                        slab = fsrc[k][s:e]            # sequential read
                        for arm, li, olo, ohi in todo:
                            dsets[arm][k][olo:ohi] = slab[li]
            print(f"  copied {os.path.basename(path)} ({time.time() - t0:.1f}s)", flush=True)
    finally:
        for fo in handles.values():
            fo.close()

    out = {}
    for arm in selections:
        nlinks = make_test_symlinks(os.path.join(OUT_ROOT, arm, "test"))
        out[arm] = (paths[arm], os.path.getsize(paths[arm]), nlinks)
    print(f"  copy pass done in {time.time() - t0:.1f}s")
    return out, time.time() - t0


def main():
    files = pool_files()
    resolve_array_keys(files)
    offsets = verify_counts(files)
    n_total = int(offsets[-1])

    Z, diag = load_inputs(files, n_total)
    if DIAG_KEY is not None:
        hot = np.abs(diag) >= np.quantile(np.abs(diag), 0.99)   # pool top-1% |OUT_Q_ions|
        print(f"top-1% |{DIAG_KEY}| cut = {np.quantile(np.abs(diag), 0.99):.6g} "
              f"({hot.sum()} rows, {100 * hot.mean():.3f}% of pool)")
    else:
        hot = np.zeros(n_total, dtype=bool)

    os.makedirs(OUT_ROOT, exist_ok=True)
    wall = time.time()
    selections, stats = {}, {}
    for arm in ARMS:
        print(f"\n=== arm {arm} ===", flush=True)
        rng = np.random.default_rng(SEED)   # same seed per arm -> arms differ only by construction
        t_sel = time.time()
        if arm == "random":
            sel, w = sel_random(n_total, rng)
        elif arm == "lightweight":
            sel, w = sel_lightweight(Z, rng)
        elif arm == "leverage_rff":
            sel, w = sel_leverage_rff(Z, rng)
        else:
            raise SystemExit(f"unknown arm {arm}")
        assert sel.min() >= 0 and sel.max() < n_total
        assert np.all(np.isfinite(w)) and np.all(w > 0)
        t_sel = time.time() - t_sel
        # With-replacement draws repeat rows. Storing a row once with the SUM of
        # its draws' weights is exactly equivalent (sum_j w_j over the duplicate
        # draws is what enters sum(w*se)), keeps total mass identical, and keeps
        # the index array strictly ascending -- which h5py fancy indexing and
        # the copy planner both require.
        n_draws = len(sel)
        sel_u, inv = np.unique(sel, return_inverse=True)
        w_u = np.bincount(inv, weights=w)
        assert abs(w_u.sum() - w.sum()) < 1e-6 * max(1.0, w.sum())
        if len(sel_u) != n_draws:
            print(f"  collapsed {n_draws} draws -> {len(sel_u)} unique rows "
                  f"(total weight preserved: {w_u.sum():.6g})")
        sel, w = sel_u.astype(np.int64), w_u
        frac_hot = float(hot[sel].mean()) if DIAG_KEY is not None else float("nan")
        print(f"  selection: {K} rows in {t_sel:.1f}s, w in "
              f"[{w.min():.4g}, {w.max():.4g}], sum(w)={w.sum():.6g}")
        if DIAG_KEY is not None:
            print(f"  top-1% |{DIAG_KEY}| share: {100 * frac_hot:.3f}%  "
                  f"(enrichment {frac_hot / 0.01:.2f}x)", flush=True)
        selections[arm] = (sel, w)
        stats[arm] = (t_sel, frac_hot)

    del Z
    print(f"\n=== copying rows (one sequential pass, {len(ARMS)} arms at once) ===", flush=True)
    written, t_io = write_all_arms(selections, files, offsets)

    print("\n==================== SUMMARY ====================")
    print(f"{'arm':<14}{'sel s':>8}{'size GB':>10}{'top1% Qi':>11}{'enrich':>9}  path")
    for arm in ARMS:
        t_sel, frac_hot = stats[arm]
        path, nbytes, nlinks = written[arm]
        print(f"{arm:<14}{t_sel:>8.1f}{nbytes / 1e9:>10.2f}"
              f"{100 * frac_hot:>10.2f}%{frac_hot / 0.01:>8.2f}x  {path} (+{nlinks} test symlinks)")
    print(f"copy pass {t_io:.1f}s, total wall {time.time() - wall:.1f}s -> {OUT_ROOT}")


if __name__ == "__main__":
    sys.exit(main())
