"""Pull the Phase-2 CGYRO bootstrap BAL runs to JSON, aligned and audited.

PHASE 2 ONLY. ``src/bal_finetune.py:113`` logs to ``CGYRO-train-fixed-op``;
``src/finetuning.py:35`` logs to ``CGYRO-eval``. Phase 1 (the fixed-data
calibration) is aggregated from the LOCAL driver's JSON
(``test/bootstrap_fixed_results.json``, written by ``test/run_bootstrap_fixed.py``)
because that record exists even with ``board=False`` and is authoritative. Do not
try to join the two phases through one wandb project.

Two encoded traps carried over from ``test/pull_acq_sweep.py``:

* **One ``history()`` call per metric.** wandb drops any row missing ANY requested
  key, so bundling ``BAL/total_samples`` with ``BAL/test_loss`` silently returns
  an empty history.
* **The off-by-one.** ``BAL/total_samples`` is logged AFTER each acquisition while
  ``BAL/test_loss`` is logged before it, so ``loss[i]`` pairs with ``nsamp[i-1]``,
  and ``loss[0]`` -- the pretrained model on the initial train set -- pairs with
  ``nsamp[0] - nnew[0]``. ``align()`` does this once, at pull time, and emits a
  ready-to-use ``points`` list so no downstream script can get it wrong.
  Sanity target from the pre-existing sweep: ``random`` = 1.8053 at 1691 samples
  (``test/cgyro_acq_sweep_all.md``).

The audit assertions below REFUSE to write the JSON. They are cheap and each one
guards a way this study gets quietly ruined -- most of all (b): pods
``git reset --hard origin/lucas_runs``, and the dropout fix will land on that
branch during the study, so one relaunched pod can mix code versions.

Needs network. Run it yourself; nothing here should be run unattended.
"""
import os
import re
import json
from collections import defaultdict

import wandb

# === USER INPUT ===
PROJECT       = "CGYRO-train-fixed-op"
ENTITY        = None              # None -> wandb default entity
AFTER         = "2026-09-08T17:40"  # ISO prefix; runs created before this are skipped.
                                  # 17:40 = just before the post-145b2bb relaunch, whose
                                  # pods came up at 17:52 and started their FIRST arm
                                  # immediately. Do NOT round this up to 18:00: the first
                                  # arm in each lane (random, strat_uni_pflip, res_uni_ran)
                                  # is created within minutes of pod start, so a later
                                  # cutoff silently drops the lane leader on every split --
                                  # including `random`, the baseline every delta uses.
                                  # Everything before this is PRE-train/-dir-fix and
                                  # contaminated (PROJECT_KNOWLEDGE.md 15.10); the earlier
                                  # 15x30/epochs=300 study is archived in
                                  # test/bootstrap_data_15x30.json -- do NOT mix them.
EXPECT_ACQS   = [# ALL 36 registered strategies, complete on all 10 splits as of
                 # 2026-09-16 (waves 1-6 + the two mop-ups). 360/360 cells.
                 # waves 1-3
                 "random", "rho_uni_ran", "strat_uni_ran", "glo_uni_lcmd",
                 "glo_uni_pflip", "eig",
                 "strat_uni_pflip", "res_uni_pflip", "strat_uni_lcmd",
                 "res_uni_lcmd", "res_rank_ran", "res_rank_pflip",
                 "res_uni_ran", "glo_uni_ow", "strat_uni_ow", "res_uni_ow",
                 "res_rank_ow", "res_uni_eig",
                 # wave 4 -- the Rho row
                 "rho_uni_pflip", "rho_uni_ow", "rho_uni_lcmd",
                 "rho_rank_ran", "rho_rank_ow", "rho_uni_eig",
                 # wave 5 (+ mop-up) -- the Ranked/EIG budget axis
                 "strat_rank_ran", "strat_rank_ow", "strat_rank_lcmd",
                 "res_rank_lcmd", "rho_rank_lcmd", "res_rank_eig",
                 # wave 6 -- the last six
                 "direct", "rho_rank_pflip", "strat_eig_pflip",
                 "strat_uni_eig", "eig_stratified", "rho_rank_eig"]
# 30 acquisition rounds -> up to 31 points. Arms exhaust the ~1690 unique-hash
# pool at slightly different iterations (strategies that dedupe more run longer),
# so requiring all 31 would drop whole splits under the complete-case rule.
EXPECT_ITERS  = 28
MANIFEST_ROOT = "./data_cgyro_boot"
OUT           = "./test/bootstrap_data.json"
BASELINE_ACQ  = "random"
# loss[0] is the SAME model on the SAME data for every arm, so any spread across
# arms within a split is pure measurement noise (nondeterministic GPU reductions
# on heterogeneous Nautilus nodes -- 4 of 6 arms come back bit-identical). It is
# NOT evidence that pairing broke: a genuine break would move loss[0] by
# sigma_split ~ 0.35. Fail only well below any effect we could claim.
LOSS0_TOL     = 0.05
# ==================

SPLIT_RE = re.compile(r"split_(\d+)")


def _hist(run, key):
    """One history() call per metric -- see the module docstring."""
    out = {}
    for h in run.history(keys=["BAL/iteration", key], pandas=False):
        it, v = h.get("BAL/iteration"), h.get(key)
        if it is not None and v is not None:
            out[int(it)] = v
    return out


def align(loss, nsamp, nnew):
    """Pair each loss with the train-set size the model was trained on.

    total_samples[i] is logged AFTER iteration i's acquisition; test_loss[i] is
    logged before it. So loss[i] <-> nsamp[i-1], and loss[0] (pretrained model on
    the initial set) <-> nsamp[0] - nnew[0].
    """
    points = []
    for it in sorted(loss):
        if it == 0:
            if 0 not in nsamp or 0 not in nnew:
                continue
            n = int(nsamp[0]) - int(nnew[0])
        else:
            if (it - 1) not in nsamp:
                continue
            n = int(nsamp[it - 1])
        points.append(dict(iteration=int(it), n=int(n), loss=float(loss[it])))
    return points


def load_manifests(root):
    mans = {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, name, "manifest.json")
        if os.path.exists(p):
            m = json.load(open(p))
            mans[int(m["split_id"])] = m
    return mans


def main():
    api = wandb.Api()
    path = PROJECT if ENTITY is None else f"{ENTITY}/{PROJECT}"

    raw = []
    for r in api.runs(path, order="-created_at", per_page=100):
        if str(r.created_at) < AFTER:
            continue
        c = r.config
        bal = c.get("bal") or {}
        acq = bal.get("acquisition_function")
        root = ((c.get("dataset") or {}).get("dataset_root")) or ""
        m = SPLIT_RE.search(str(root))
        if acq is None or m is None:
            continue
        split_id = int(m.group(1))

        loss = {k: float(v) for k, v in _hist(r, "BAL/test_loss").items()}
        nsamp = {k: int(v) for k, v in _hist(r, "BAL/total_samples").items()}
        nnew = {k: int(v) for k, v in _hist(r, "BAL/num_samples").items()}

        raw.append(dict(
            name=r.name, id=r.id, state=r.state, created_at=str(r.created_at),
            acq=acq, split_id=split_id, dataset_root=root,
            seed=c.get("base_seed"), epochs=c.get("epochs"),
            peak_lr=(c.get("opt") or {}).get("peak_lr"),
            weight_decay=(c.get("opt") or {}).get("weight_decay"),
            new_sample_size=bal.get("new_sample_size"),
            initial_training_size=bal.get("initial_training_size"),
            n_samples=bal.get("n_samples"),
            iterations=bal.get("iterations"),
            git_commit=(r.metadata or {}).get("git", {}).get("commit")
                       if isinstance(r.metadata, dict) else None,
            loss=loss, nsamp=nsamp, nnew=nnew,
            points=align(loss, nsamp, nnew),
        ))
        print(f"{acq:16s} split={split_id:<3d} {r.state:9s} "
              f"iters={len(loss)} pts={len(raw[-1]['points'])}")

    assert raw, f"no matching runs in {path} after {AFTER}"

    # ---- dedupe on (repeat, fold, acq, seed) --------------------------------
    mans = load_manifests(MANIFEST_ROOT)
    assert mans, f"no manifests under {MANIFEST_ROOT}"

    def key(rec):
        man = mans.get(rec["split_id"])
        rep = man["repeat"] if man else -1
        fold = man["fold"] if man else -1
        return (rep, fold, rec["acq"], rec["seed"])

    groups = defaultdict(list)
    for rec in raw:
        groups[key(rec)].append(rec)

    runs, dropped = [], []
    for k, recs in groups.items():
        recs.sort(key=lambda r: (len(r["points"]), r["created_at"]), reverse=True)
        runs.append(recs[0])
        for r in recs[1:]:
            dropped.append((k, r["name"], r["id"], len(r["points"]), r["created_at"]))
    if dropped:
        print(f"\ndeduped {len(dropped)} run(s) on (repeat,fold,acq,seed):")
        for k, name, rid, npts, ts in dropped:
            print(f"  DROPPED {k} {name} id={rid} pts={npts} created={ts}")

    # ---- audit -------------------------------------------------------------
    problems = []

    # (a) loss[0] agrees across acq arms within a split. Iteration 0 is the
    #     pretrained model trained on the initial train set, BEFORE any
    #     acquisition, so every arm sees identical data -- it is the cheapest
    #     proof that pairing holds. Its residual spread is a free, direct
    #     measurement of sigma_meas: same data, same seed, different pod.
    by_split = defaultdict(dict)
    for r in runs:
        if 0 in r["loss"]:
            by_split[r["split_id"]][r["acq"]] = r["loss"][0]
    print("\n=== (a) loss[0] agreement -- free sigma_meas estimate ===")
    print(f"{'split':>5s} {'mean':>10s} {'spread':>10s} {'sd':>10s} {'identical':>10s}")
    spreads, sds, noisy = [], [], []
    for sid, arms in sorted(by_split.items()):
        vals = list(arms.values())
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        nmode = max(sum(1 for v in vals if round(v, 12) == round(u, 12)) for u in vals)
        spreads.append(spread)
        sds.append(sd)
        print(f"{sid:5d} {mean:10.4f} {spread:10.5f} {sd:10.5f} "
              f"{nmode:>7d}/{len(vals)}")
        if spread > LOSS0_TOL:
            noisy.append(sid)
    if spreads:
        gs = (sum(s * s for s in sds) / len(sds)) ** 0.5
        print(f"\n  max spread {max(spreads):.5f}  (diagnostic tol {LOSS0_TOL})")
        print(f"  pooled sigma_meas = {gs:.5f}   -> paired sigma_D ~ {gs * 2 ** 0.5:.5f}")
        if noisy:
            print(f"  NOTE: splits {noisy} exceed the tol. This is NOT a pairing\n"
                  f"        failure -- see (a2), which tests pairing directly.\n"
                  f"        BAL/test_loss averages per-BATCH means over a reshuffled\n"
                  f"        loader (bal_finetune.py:100-106, batch=256), so a single\n"
                  f"        point carries ~0.18 sd of pure estimator noise at iter 0.\n"
                  f"        Fix is batch_size=10**6 as src/finetuning.py:163-174 does.")

    # (a2) THE REAL PAIRING TEST -- identical initial training set per split.
    # base_seed is constant and the initial draw precedes any acquisition, so
    # every arm on a split must start from the same rows. Size is the observable
    # proxy; it is post-dedup, hence sensitive to the actual row content.
    print("\n=== (a2) initial train size agreement (the real pairing test) ===")
    init = defaultdict(dict)
    for r in runs:
        if r["points"]:
            init[r["split_id"]][r["acq"]] = r["points"][0]["n"]
    for sid, arms in sorted(init.items()):
        vals = set(arms.values())
        print(f"{sid:5d}  n={sorted(vals)}  arms={len(arms)}")
        if len(vals) > 1:
            problems.append(f"(a2) split {sid}: initial train size differs across "
                            f"arms {arms} -- pairing IS broken")

    # (b) single git commit across the study
    commits = {r["git_commit"] for r in runs if r["git_commit"]}
    if len(commits) > 1:
        problems.append(f"(b) runs span {len(commits)} git commits: {sorted(commits)}")
    elif not commits:
        print("(b) WARNING: no git commit recorded in run metadata; cannot check")

    # (c) constant hyperparameters
    for field in ("epochs", "peak_lr", "weight_decay", "new_sample_size",
                  "initial_training_size", "n_samples", "iterations", "seed"):
        vals = {r[field] for r in runs}
        if len(vals) > 1:
            problems.append(f"(c) {field} is not constant: {sorted(map(str, vals))}")

    # (d) test row + valid-ky counts match the manifests
    for r in runs:
        man = mans.get(r["split_id"])
        if man is None:
            problems.append(f"(d) split {r['split_id']}: no manifest under {MANIFEST_ROOT}")
            continue
        r["test_rows"] = man["test"]["rows"]
        r["test_valid_ky"] = man["test"]["valid_ky"]
        r["pool_rows"] = man["train"]["rows"]
        r["n_pool_groups"] = man["n_pool_groups"]
        r["repeat"], r["fold"] = man["repeat"], man["fold"]
        r["split_fingerprint"] = man["split_fingerprint"]
        if man["leaked_test_rows"] != 0:
            problems.append(f"(d) split {r['split_id']}: manifest reports "
                            f"{man['leaked_test_rows']} leaked test rows")
    fps = {mans[r["split_id"]]["source_fingerprint"] for r in runs if r["split_id"] in mans}
    if len(fps) > 1:
        problems.append(f"(d) manifests span {len(fps)} source fingerprints: {fps}")

    # (e) completion matrix
    splits = sorted({r["split_id"] for r in runs})
    grid = {(r["split_id"], r["acq"]): len(r["points"]) for r in runs}
    print("\n=== completion matrix (points per run; want "
          f"{EXPECT_ITERS + 1}) ===")
    print("split " + "".join(f"{a[:13]:>15s}" for a in EXPECT_ACQS))
    for sid in splits:
        row = "".join(f"{grid.get((sid, a), 0):>15d}" for a in EXPECT_ACQS)
        print(f"{sid:5d} {row}")
    complete = [s for s in splits
                if all(grid.get((s, a), 0) >= EXPECT_ITERS for a in EXPECT_ACQS)]
    print(f"\ncomplete-case splits (all {len(EXPECT_ACQS)} arms, >= {EXPECT_ITERS} "
          f"points): {len(complete)} -> {complete}")
    missing = [(s, a) for s in splits for a in EXPECT_ACQS if (s, a) not in grid]
    if missing:
        print(f"missing cells: {missing}")
    if BASELINE_ACQ not in EXPECT_ACQS:
        problems.append(f"(e) BASELINE_ACQ {BASELINE_ACQ!r} not in EXPECT_ACQS")

    if problems:
        print("\n!! AUDIT FAILED -- refusing to write " + OUT)
        for p in problems:
            print("  " + p)
        raise SystemExit(1)

    payload = dict(project=PROJECT, after=AFTER, baseline=BASELINE_ACQ,
                   expect_acqs=EXPECT_ACQS, expect_iters=EXPECT_ITERS,
                   complete_case_splits=complete, runs=runs,
                   dropped=[list(map(str, d)) for d in dropped])
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"\naudit clean; wrote {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    main()
