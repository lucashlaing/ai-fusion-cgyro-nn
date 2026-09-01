"""Stratified GROUP K-fold bootstrap resplits of the CGYRO rho dataset.

Why this exists
---------------
``test/split_test_train.py`` shuffles per ROW. The CGYRO rho dataset holds 2778
rows but only 2112 unique physics groups under the repo's own fingerprint (SHA1
of the 31 input columns as float32, ``src/utils/Tracker.py:10-18``), and all 255
multi-row groups span more than one source file. A per-row split therefore tears
groups apart: ``./data_cgyro_split`` leaks 156 of its 556 test rows (28.1%).

This script splits on the physics hash, so leakage is zero by construction, and
produces ``N_REPEATS`` independent repeats of a K-fold partition -> ``N_REPEATS *
K_FOLDS`` splits that can be used to compare acquisition functions *paired on
identical splits*, which is the only way to cancel the enormous CGYRO split
variance (seed spread 0.31 vs acquisition spread 0.045).

Two details that are load-bearing:

* **Folds are balanced on VALID-KY count, not rows.** ``meta/failed_mask`` drops
  32.7% of ky cells and survival is strongly rho-dependent (11.9 valid ky/row at
  rho=0.2 vs 19.2 at rho=0.9). The loss is a mean over surviving ky rows, so
  balancing rows leaves a 12% spread in the actual loss denominator; balancing
  valid-ky leaves 0.15%.
* **The greedy bin-packer breaks ties at RANDOM, not with ``np.argmin``.**
  ``argmin`` always returns fold 0, which correlates the repeats: measured
  cross-repeat test-set overlap climbs to 211 rows against a chance value of
  ~109. A random tie-break lands at ~113.

Reproducibility contract
------------------------
The partition is derived from ``SEED`` alone -- nothing is stored. Running this
script anywhere with the same seed and the same source data reproduces the same
40 splits, which is what lets a pod materialize the exact partitions the Phase-1
calibration was measured on. Verified: the ``default_rng([SEED, repeat])`` /
``permutation`` / ``choice`` stream is byte-identical under numpy 2.1.0 (local)
and 2.3.1 (poetry.lock), and the image ends on 2.1.0 anyway because pyrokinetics
pins ``numpy<=2.1.0``.

The one thing a seed cannot protect against is the SOURCE DATA changing, so
``EXPECT_SOURCE_FINGERPRINT`` is checked on every run and raises on mismatch --
five files were added to this dataset on 2026-08-23, which would silently
redefine every split. Changing ``assign_folds`` likewise changes the partition;
that invalidates the calibration, which must then be re-run with
``test/run_bootstrap_fixed.py``.

Layouts are always MATERIALIZED, never symlinked: BAL writes ``initial_train.h5``
/ ``BAL_<i>_new.h5`` / ``candidates.h5`` / ``temp_entropy_data.h5`` into
``<root>/train``, so a ``train -> pool`` symlink would permanently contaminate
the split dir.

Every constant below is overridable by the environment variable ``BOOT_<NAME>``.
"""
import os
import glob
import json
import hashlib
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
import sys

# self_check step 7 imports the repo's datapipe. Pods get PYTHONPATH=src from
# kubeutils.py:395-398, but a bare local `python test/make_bootstrap_splits.py`
# does not -- and it crashed the check AFTER the h5s were written, leaving a
# half-verified split. Same pattern as test/save_predictions.py:11.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import h5py
import numpy as np


# === USER INPUT (env override: BOOT_<NAME>) ===
SRC_DIR      = "/data/lucas_work/cgyro-rho-sep/rho"
OUT_ROOT     = "./data_cgyro_boot"
K_FOLDS      = 5
N_REPEATS    = 8
SEED         = 20260831
LAYOUT       = "fixed"      # fixed | bal
ONLY         = ""           # "" -> all splits; else comma-separated split ids
SELF_CHECK   = True
# The partition is derived from SEED alone -- there is no stored index. The one
# thing a seed cannot protect against is the SOURCE DATA changing: five files were
# added to this dataset on 2026-08-23, which would silently redefine every split.
# Set to "" to bypass after a deliberate dataset change (and re-run the base case).
EXPECT_SOURCE_FINGERPRINT = "7efd43ad7ab9a016975e6f9173dac191"
# ==============================================

SCHEMA_VERSION = 1
TRAIN_NAME = "all_train.h5"
TEST_NAME  = "all_test.h5"
POOL_NAME  = "all_pool.h5"
DATASET_CFG = "run_configs/dataset/CGYRO_local.yaml"

# The 31 physics inputs in the EXACT order of run_configs/dataset/CGYRO_local.yaml.
# Order determines the SHA1 group hash, so it is asserted against the yaml below.
INPUT_KEYS = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1", "RLTS_2",
    "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3", "RLNS_3",
    "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR", "RMAJ_LOC",
    "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10",
]

# Names as stored in the CGYRO h5s (NOT the yaml target_keys, which are the
# TGLF-style OUT_G_elec/... -- the reconcile check in CGYRO_Spectra.py:148 uses these).
OUT_KEYS = ["OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i"]
MASK_KEY = "meta/failed_mask"


# ----------------------------------------------------------------------
# env overrides
# ----------------------------------------------------------------------
def _env_override():
    g = globals()
    for name in ("SRC_DIR", "OUT_ROOT", "K_FOLDS", "N_REPEATS", "SEED",
                 "LAYOUT", "ONLY", "SELF_CHECK", "EXPECT_SOURCE_FINGERPRINT"):
        raw = os.environ.get("BOOT_" + name)
        if raw is None:
            continue
        cur = g[name]
        if isinstance(cur, bool):
            g[name] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            g[name] = int(raw)
        else:
            g[name] = raw
    assert LAYOUT in ("fixed", "bal"), f"bad BOOT_LAYOUT={LAYOUT!r}"


def _assert_input_keys():
    """INPUT_KEYS must equal the yaml's input_keys, in order -- it is the hash."""
    from omegaconf import OmegaConf
    if not os.path.exists(DATASET_CFG):
        print(f"WARNING: {DATASET_CFG} not found; skipping INPUT_KEYS assert")
        return
    yaml_keys = list(OmegaConf.load(DATASET_CFG).input_keys)
    assert yaml_keys == INPUT_KEYS, (
        "INPUT_KEYS drifted from {}:\n  yaml: {}\n  here: {}".format(
            DATASET_CFG, yaml_keys, INPUT_KEYS)
    )


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def list_source_files():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.h5")))
    files = [p for p in files
             if os.path.basename(p) not in {TRAIN_NAME, TEST_NAME, POOL_NAME}]
    assert files, f"no h5 files under {SRC_DIR}"
    return files


def source_fingerprint(files):
    """md5 of the sorted BASENAMES joined by newline -- path-independent."""
    joined = "\n".join(sorted(os.path.basename(p) for p in files)) + "\n"
    return hashlib.md5(joined.encode()).hexdigest()


def config_hash():
    payload = json.dumps(dict(schema_version=SCHEMA_VERSION, seed=SEED,
                              k_folds=K_FOLDS, n_repeats=N_REPEATS,
                              input_keys=INPUT_KEYS), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def discover_schema(path):
    """name -> (inner_shape, dtype) for every dataset, nested groups included."""
    schema = {}
    with h5py.File(path, "r") as f0:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                schema[name] = (obj.shape[1:], obj.dtype)
        f0.visititems(visit)
    return schema


def load_all(files):
    """Pool every dataset across files, truncating each file to its min leading dim."""
    schema = discover_schema(files[0])
    buffers = {name: [] for name in schema}
    per_file_counts = []
    provenance = []   # (file_idx, local_row) per pooled row

    for fi, path in enumerate(files):
        with h5py.File(path, "r") as f:
            missing = [k for k in schema if k not in f]
            assert not missing, f"{path}: missing datasets {missing}"
            leading = [f[name].shape[0] for name in schema]
            N = int(min(leading))
            per_file_counts.append([os.path.basename(path), N, int(max(leading))])
            for name in schema:
                buffers[name].append(f[name][:N])
            provenance.extend((fi, r) for r in range(N))

    data = {name: np.concatenate(arrs, axis=0) for name, arrs in buffers.items()}
    n_total = next(iter(data.values())).shape[0]
    assert all(a.shape[0] == n_total for a in data.values()), "ragged pooled counts"
    return data, schema, per_file_counts, np.asarray(provenance, dtype=np.int64), n_total


def physics_hashes(data):
    """Byte-identical to src/utils/Tracker.py:10-18, on raw h5 values.

    Safe: log_ops_keys is applied only by src/simulator/h5_to_cgyro_input.py,
    never by a datapipe, so these match what UsageTracker sees.
    """
    arr = np.stack([data[k] for k in INPUT_KEYS], axis=1).astype(np.float32)
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in arr]


def valid_ky_per_row(data):
    return (np.asarray(data[MASK_KEY]) == 0).sum(axis=1).astype(np.int64)


def rho_bands(data):
    return np.round(np.asarray(data["rho"], dtype=np.float64), 1)


# ----------------------------------------------------------------------
# fold assignment
# ----------------------------------------------------------------------
def assign_folds(hashes, bands, weights, repeat):
    """Stratified group K-fold via longest-processing-time greedy bin packing.

    Per rho band, groups (by physics hash) are shuffled into a canonical random
    order, sorted by descending valid-ky weight (stable, so the shuffle survives
    inside ties), then dropped one at a time into the currently lightest fold.
    Ties on load are broken at RANDOM -- np.argmin here is a real bug that
    correlates repeats (overlap 211 vs ~109 chance).
    """
    n = len(hashes)
    fold = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng([SEED, repeat])

    for band in sorted(set(bands.tolist())):
        band_rows = np.flatnonzero(bands == band)
        groups = defaultdict(list)
        for r in band_rows:
            groups[hashes[r]].append(int(r))

        items = sorted(groups.items(), key=lambda kv: kv[0])          # canonical
        items = [(h, np.asarray(rows, dtype=np.int64),
                  int(weights[rows].sum())) for h, rows in items]
        order = rng.permutation(len(items))                           # shuffle
        items = [items[i] for i in order]
        items.sort(key=lambda it: -it[2])                             # LPT (stable)

        load = np.zeros(K_FOLDS, dtype=np.int64)
        for _h, rows, w in items:
            cands = np.flatnonzero(load == load.min())
            f = int(rng.choice(cands))
            load[f] += w
            fold[rows] = f

    assert (fold >= 0).all(), "some rows left unassigned"
    return fold


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def write_h5(out_path, data, idx):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as out:
        for name, arr in data.items():
            sub = arr[idx]
            if "/" in name:
                grp_name, ds_name = name.rsplit("/", 1)
                out.require_group(grp_name).create_dataset(ds_name, data=sub)
            else:
                out.create_dataset(name, data=sub)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def half_stats(idx, hashes, bands, weights):
    per_rho_rows, per_rho_vk = {}, {}
    for band in sorted(set(bands.tolist())):
        sel = idx[bands[idx] == band]
        per_rho_rows[f"{band:.1f}"] = int(len(sel))
        per_rho_vk[f"{band:.1f}"] = int(weights[sel].sum())
    return dict(
        rows=int(len(idx)),
        groups=int(len({hashes[i] for i in idx})),
        valid_ky=int(weights[idx].sum()),
        per_rho_rows=per_rho_rows,
        per_rho_validky=per_rho_vk,
    )


def materialize_split(split_id, assign_str, repeat, fold, data, hashes, bands,
                      weights, provenance, files, per_file_counts, src_fp, n_out_nonfinite):
    assign = np.frombuffer(assign_str.encode(), dtype=np.uint8) - ord("0")
    assign = assign.astype(np.int64)
    assert len(assign) == len(hashes), "index length != source row count"

    test_idx = np.flatnonzero(assign == fold)
    train_idx = np.flatnonzero(assign != fold)

    split_dir = os.path.join(OUT_ROOT, f"split_{split_id}")
    os.makedirs(split_dir, exist_ok=True)

    if LAYOUT == "fixed":
        big_path = os.path.join(split_dir, "train", TRAIN_NAME)
    else:
        big_path = os.path.join(split_dir, "pool", POOL_NAME)
        os.makedirs(os.path.join(split_dir, "train"), exist_ok=True)
    test_path = os.path.join(split_dir, "test", TEST_NAME)

    write_h5(big_path, data, train_idx)
    write_h5(test_path, data, test_idx)

    train_h = {hashes[i] for i in train_idx}
    leaked_rows = int(sum(1 for i in test_idx if hashes[i] in train_h))

    manifest = dict(
        split_id=split_id, repeat=repeat, fold=fold, layout=LAYOUT,
        seed=SEED, k_folds=K_FOLDS, n_repeats=N_REPEATS,
        train=half_stats(train_idx, hashes, bands, weights),
        test=half_stats(test_idx, hashes, bands, weights),
        leaked_test_rows=leaked_rows,
        n_pool_groups=int(len(train_h)),
        n_nonfinite_out_rows=int(n_out_nonfinite),
        source_files=[os.path.basename(p) for p in files],
        per_file_counts=per_file_counts,
        source_fingerprint=src_fp,
        config_hash=config_hash(),
        split_fingerprint=hashlib.sha1(
            (src_fp + assign_str + str(fold)).encode()).hexdigest(),
        git_commit=git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        _train_provenance=provenance[train_idx].tolist(),
        _test_provenance=provenance[test_idx].tolist(),
    )
    with open(os.path.join(split_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    return split_dir, manifest


# ----------------------------------------------------------------------
# self-check
# ----------------------------------------------------------------------
def _read_hashes(path):
    with h5py.File(path, "r") as f:
        arr = np.stack([np.asarray(f[k]) for k in INPUT_KEYS], axis=1).astype(np.float32)
    return [hashlib.sha1(r.tobytes()).hexdigest() for r in arr]


def leak_report(train_path, test_path):
    """Standalone leak checker; also runnable against ./data_cgyro_split."""
    tr = set(_read_hashes(train_path))
    te = _read_hashes(test_path)
    leaked = [h for h in te if h in tr]
    return len(leaked), len(te), len(set(leaked))


def _derive_targets(sumf):
    """CGYRO axes, mirroring src/dataset/CGYRO_Spectra.py:59-62."""
    fx = sumf[:, :, 0]                       # (N, nky, ns, nf, 5)
    fx = fx.sum(axis=3)                      # sum FIELD -> (N, nky, ns, 5)
    ge = fx[:, :, 0, 0]
    qe = fx[:, :, 0, 1]
    qi = fx[:, :, 1:, 1].sum(axis=-1)
    pi = fx[:, :, 1:, 2].sum(axis=-1)
    return [a.sum(axis=1) for a in (ge, qe, qi, pi)]


def self_check(split_dir, manifest, layout, files, n_total):
    from omegaconf import OmegaConf, open_dict

    big_name = TRAIN_NAME if layout == "fixed" else POOL_NAME
    big_dir = "train" if layout == "fixed" else "pool"
    big_path = os.path.join(split_dir, big_dir, big_name)
    test_path = os.path.join(split_dir, "test", TEST_NAME)

    tr_h = _read_hashes(big_path)
    te_h = _read_hashes(test_path)

    # 1. conservation
    assert len(tr_h) + len(te_h) == n_total, \
        f"{split_dir}: {len(tr_h)}+{len(te_h)} != {n_total}"
    from collections import Counter
    src_h = self_check.src_hashes
    assert Counter(tr_h) + Counter(te_h) == Counter(src_h), \
        f"{split_dir}: hash multiset != source multiset"

    # 2. zero leak, recomputed from the WRITTEN files
    inter = set(tr_h) & set(te_h)
    assert not inter, f"{split_dir}: LEAK -- {len(inter)} groups in both halves"

    # 3. all 9 rho bands present both halves
    with h5py.File(big_path, "r") as f:
        tr_rho = np.round(np.asarray(f["rho"], dtype=np.float64), 1)
        tr_sumf_shape = f["sumf"].shape
    with h5py.File(test_path, "r") as f:
        te_rho = np.round(np.asarray(f["rho"], dtype=np.float64), 1)
    bands = sorted(set(np.round(self_check.src_bands, 1).tolist()))
    assert len(bands) == 9, f"expected 9 rho bands, saw {bands}"
    for b in bands:
        assert (tr_rho == b).sum() > 0, f"{split_dir}: rho {b} missing from {big_dir}"
        assert (te_rho == b).sum() > 0, f"{split_dir}: rho {b} missing from test"

    # 4. test valid-ky within 1% of total/K
    target = self_check.total_valid_ky / K_FOLDS
    got = manifest["test"]["valid_ky"]
    assert abs(got - target) / target <= 0.01, \
        f"{split_dir}: test valid_ky {got} off target {target:.1f} by >1%"

    # 5. target re-derivation from the written sumf
    for path in (big_path, test_path):
        with h5py.File(path, "r") as f:
            sumf = np.asarray(f["sumf"], dtype=np.float64)
            outs = [np.asarray(f[k], dtype=np.float64) for k in OUT_KEYS]
        der = _derive_targets(sumf)
        finite = np.ones(len(sumf), dtype=bool)
        for o in outs:
            finite &= np.isfinite(o)
        bad = 0
        for d, o in zip(der, outs):
            bad += int((~np.isclose(d[finite], o[finite], rtol=1e-3,
                                    atol=1e-30)).sum())
        assert bad == 0, f"{path}: {bad} target re-derivation disagreements"

    # 6. byte-level row alignment against the sources, via provenance.
    # (This fixed-seed rng only chooses WHICH rows to verify; it touches no
    # output, so the "no RNG on the materialize path" rule still holds for the
    # data itself -- the written h5s are byte-identical with SELF_CHECK on/off.)
    rng = np.random.default_rng(0)
    schema = self_check.schema
    for half, prov_key, path in (("train", "_train_provenance", big_path),
                                 ("test", "_test_provenance", test_path)):
        prov = manifest[prov_key]
        if not prov:
            continue
        pick = rng.choice(len(prov), size=min(20, len(prov)), replace=False)
        with h5py.File(path, "r") as fo:
            for oi in pick:
                fidx, lrow = prov[int(oi)]
                with h5py.File(files[fidx], "r") as fs:
                    for name in schema:
                        a = np.asarray(fo[name][int(oi)])
                        b = np.asarray(fs[name][int(lrow)])
                        assert np.array_equal(a, b), (
                            f"{path}: row {oi} dataset {name} != "
                            f"{os.path.basename(files[fidx])}[{lrow}]")

    # 7. datapipe load
    from dataset.CGYRO_Spectra import CGYRO_Spectra_DataPipe
    dcfg = OmegaConf.load(DATASET_CFG)
    with open_dict(dcfg):
        dcfg.dataset_root = split_dir
    modes = ("train", "test") if layout == "fixed" else ("pool", "test")
    for mode in modes:
        pipe = CGYRO_Spectra_DataPipe(dcfg, 1, 42, mode)
        n_rows, n_ky = 0, 0
        for inp, tgt in pipe:
            n_rows += 1
            n_ky += int(inp.shape[0])
        key = "train" if mode in ("train", "pool") else "test"
        assert n_rows == manifest[key]["rows"], \
            f"{split_dir}/{mode}: datapipe yielded {n_rows} rows, manifest {manifest[key]['rows']}"
        assert n_ky <= manifest[key]["valid_ky"], \
            f"{split_dir}/{mode}: {n_ky} ky rows > manifest valid_ky {manifest[key]['valid_ky']}"

    # 8. shape guard
    assert tuple(tr_sumf_shape[1:]) == (24, 2, 3, 3, 5), \
        f"{split_dir}: sumf inner shape {tr_sumf_shape[1:]} != (24,2,3,3,5)"

    # 9. layout guard
    if layout == "bal":
        tdir = os.path.join(split_dir, "train")
        assert os.path.isdir(tdir), f"{split_dir}: bal layout needs an empty train/"
        stray = glob.glob(os.path.join(tdir, "**/*"), recursive=True)
        assert not stray, f"{split_dir}: bal layout train/ not empty: {stray}"
    else:
        assert not os.path.isdir(os.path.join(split_dir, "pool")), \
            f"{split_dir}: fixed layout must not have pool/"


def derive_assignments(hashes, bands, weights):
    """Fold assignment for every repeat, derived from SEED alone.

    There is deliberately no stored index. numpy's Generator stream was measured
    identical across the local (2.1.0) and container (2.3.1) numpy versions for
    this call pattern, and the image pins 2.1.0 anyway via pyrokinetics, so
    seed + source data + assign_folds reproduce the partition exactly. If
    assign_folds is ever changed the splits change with it -- but so does the
    calibration, which has to be re-run regardless (test/run_bootstrap_fixed.py).
    """
    out = {}
    for rep_i in range(N_REPEATS):
        fold = assign_folds(hashes, bands, weights, rep_i)
        assert fold.max() < 10, "K_FOLDS > 10 breaks the digit-string form"
        out[str(rep_i)] = "".join(str(int(x)) for x in fold)
    return out


# ----------------------------------------------------------------------
# cross-split checks
# ----------------------------------------------------------------------
def cross_split_check(assigns=None, out_root=None, verbose=True):
    out_root = out_root or OUT_ROOT
    mans = {}
    for d in sorted(glob.glob(os.path.join(out_root, "split_*"))):
        p = os.path.join(d, "manifest.json")
        if os.path.exists(p):
            m = json.load(open(p))
            mans[m["split_id"]] = m
    assert mans, f"no manifests under {out_root}"

    if assigns is None:      # standalone call -- re-derive from SEED
        files = list_source_files()
        data, _schema, _pfc, _prov, _n = load_all(files)
        assigns = derive_assignments(physics_hashes(data), rho_bands(data),
                                     valid_ky_per_row(data))
    assigns = {int(k): v for k, v in assigns.items()}
    n_total = len(next(iter(assigns.values())))

    ok = True
    # fingerprints
    fps = {m["source_fingerprint"] for m in mans.values()}
    assert len(fps) == 1, f"source_fingerprint differs across splits: {fps}"
    sfps = [m["split_fingerprint"] for m in mans.values()]
    assert len(set(sfps)) == len(sfps), "duplicate split_fingerprint"

    # per-repeat coverage / disjointness, from the index (the contract)
    test_sets = {}
    for rep, s in assigns.items():
        a = np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")
        counts = np.zeros(n_total, dtype=np.int64)
        for f in range(K_FOLDS):
            t = set(np.flatnonzero(a == f).tolist())
            test_sets[(rep, f)] = t
            counts[list(t)] += 1
        assert (counts == 1).all(), f"repeat {rep}: rows not tested exactly once"
        for f1 in range(K_FOLDS):
            for f2 in range(f1 + 1, K_FOLDS):
                assert not (test_sets[(rep, f1)] & test_sets[(rep, f2)]), \
                    f"repeat {rep}: folds {f1},{f2} overlap"

    # cross-repeat overlap vs chance
    chance = n_total / (K_FOLDS ** 2)
    overlaps = []
    reps = sorted(assigns)
    for i, r1 in enumerate(reps):
        for r2 in reps[i + 1:]:
            for f1 in range(K_FOLDS):
                for f2 in range(K_FOLDS):
                    overlaps.append(len(test_sets[(r1, f1)] & test_sets[(r2, f2)]))
    ov = np.asarray(overlaps, dtype=float)
    # The MEAN overlap is identically `chance` by construction (every row is
    # tested exactly once per repeat), so it carries no information. What the
    # argmin tie-break regression does is DISPERSE the distribution: measured
    # here, random tie-break -> median 110 / max 178; np.argmin -> median 93 /
    # max 242 on the very same data. So gate on the median and the tail.
    if abs(np.median(ov) - chance) > 0.15 * chance or ov.max() > 2.0 * chance:
        ok = False
        print(f"!! WARNING: cross-repeat overlap median {np.median(ov):.1f} / max "
              f"{ov.max():.0f} vs chance {chance:.1f} -- repeats look correlated. "
              f"This is what the np.argmin tie-break regression looks like.")

    if verbose:
        print("\n=== cross_split_check ===")
        print(f"manifests: {len(mans)}   rows: {n_total}   "
              f"K={K_FOLDS} R={N_REPEATS}")
        print(f"source_fingerprint: {fps.pop()} (identical across all splits)")
        print(f"split_fingerprints: {len(set(sfps))} distinct / {len(sfps)}")
        print("every row tested exactly once per repeat: OK "
              f"({N_REPEATS}x overall)")
        print("within-repeat test folds disjoint and covering: OK")
        print(f"cross-repeat test overlap: mean {ov.mean():.1f} (== chance by "
              f"construction) median {np.median(ov):.1f} max {ov.max():.0f}  "
              f"[chance {chance:.1f}; warn if median off by >15% or max >"
              f"{2.0 * chance:.0f}]")
        tvk = np.array([m["test"]["valid_ky"] for m in mans.values()], dtype=float)
        trw = np.array([m["test"]["rows"] for m in mans.values()], dtype=float)
        print(f"test valid_ky: {tvk.min():.0f}-{tvk.max():.0f} "
              f"(spread {(tvk.max() - tvk.min()) / tvk.mean() * 100:.2f}%)")
        print(f"test rows:     {trw.min():.0f}-{trw.max():.0f} "
              f"(spread {(trw.max() - trw.min()) / trw.mean() * 100:.2f}%)")
        print(f"leaked_test_rows across all splits: "
              f"{sum(m['leaked_test_rows'] for m in mans.values())}")
        print(f"result: {'OK' if ok else 'PROBLEMS FOUND'}")
    return ok


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    _env_override()
    _assert_input_keys()

    files = list_source_files()
    src_fp = source_fingerprint(files)
    print(f"LAYOUT={LAYOUT} SRC={SRC_DIR} ({len(files)} files) fingerprint={src_fp}")
    if EXPECT_SOURCE_FINGERPRINT and src_fp != EXPECT_SOURCE_FINGERPRINT:
        raise RuntimeError(
            f"source fingerprint {src_fp} != expected {EXPECT_SOURCE_FINGERPRINT}. "
            "The source dataset changed, so every split would be redefined and the "
            "Phase-1 calibration no longer applies. Update EXPECT_SOURCE_FINGERPRINT "
            "and re-run the base case (test/run_bootstrap_fixed.py) deliberately.")

    data, schema, per_file_counts, provenance, n_total = load_all(files)
    print(f"pooled rows: {n_total}   datasets: {len(schema)}")

    hashes = physics_hashes(data)
    bands = rho_bands(data)
    weights = valid_ky_per_row(data)
    total_valid_ky = int(weights.sum())
    print(f"unique physics groups: {len(set(hashes))}   "
          f"total valid ky: {total_valid_ky} "
          f"({total_valid_ky / (n_total * data['ky'].shape[1]) * 100:.1f}%)")

    n_out_nonfinite = int((~np.all(
        [np.isfinite(data[k]) for k in OUT_KEYS], axis=0)).sum())
    print(f"rows with non-finite OUT_*: {n_out_nonfinite}")

    assignments = derive_assignments(hashes, bands, weights)

    wanted = None
    if ONLY.strip():
        wanted = {int(x) for x in ONLY.replace(",", " ").split()}

    self_check.src_hashes = hashes
    self_check.src_bands = bands
    self_check.total_valid_ky = total_valid_ky
    self_check.schema = schema

    made = []
    for rep in range(N_REPEATS):
        astr = assignments[str(rep)]
        for fold in range(K_FOLDS):
            sid = rep * K_FOLDS + fold
            if wanted is not None and sid not in wanted:
                continue
            split_dir, man = materialize_split(
                sid, astr, rep, fold, data, hashes, bands, weights,
                provenance, files, per_file_counts, src_fp, n_out_nonfinite)
            if SELF_CHECK:
                self_check(split_dir, man, LAYOUT, files, n_total)
            made.append(sid)
            print(f"  split_{sid:<3d} rep={rep} fold={fold}  "
                  f"train {man['train']['rows']:4d}r/{man['train']['groups']:4d}g/"
                  f"{man['train']['valid_ky']:5d}ky   "
                  f"test {man['test']['rows']:4d}r/{man['test']['groups']:4d}g/"
                  f"{man['test']['valid_ky']:5d}ky   "
                  f"leak={man['leaked_test_rows']}"
                  f"{'  [checked]' if SELF_CHECK else ''}")

    print(f"\nmaterialized {len(made)} split(s) under {OUT_ROOT} (layout={LAYOUT})")


if __name__ == "__main__":
    main()
