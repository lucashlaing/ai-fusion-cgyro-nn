"""Phase-1 driver: fixed-data finetune on every bootstrap split, locally.

Runs ``src/finetuning.py`` once per ``data_cgyro_boot/split_<i>`` and records the
full-test-set loss it prints. That number exists only because of the six lines
added at the end of ``run_train`` -- the ``test_loss`` in the training loop is a
single batch and is not rankable -- so a missing ``FULL TEST LOSS:`` line means
the run used stale code, and this driver refuses to guess.

Output is the authoritative Phase-1 record: ``finetuning.py`` logs to wandb
project ``CGYRO-eval`` (not ``CGYRO-train-fixed-op`` where Phase 2 lands), and
this works with ``board=False`` anyway. ``test/pull_bootstrap.py`` covers Phase 2
only; do not try to join the two phases through one wandb project.

Results are appended to OUT_JSON after EVERY split, so a crash loses nothing and
a rerun can be pointed at the remaining ids.
"""
import os
import re
import json
import subprocess
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# === USER INPUT ===
PYTHON       = "/data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/bin/python"
CONDA_LIB   = "/data/lucas_work/miniconda3/envs/ai-fusion-cgyro-nn/lib"
SPLITS_ROOT  = "./data_cgyro_boot"
SPLIT_IDS    = list(range(40))
GPUS         = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_PARALLEL = 8
# Phase-1 training config, per the plan. 6000 steps = epochs*steps_per_epoch is
# the measured optimum for ~2200 rows (PROJECT_KNOWLEDGE.md 13.4: 2000 -> 1.5106,
# 6000 -> 1.2413, 18000 overfits). opt.decay_steps MUST equal the total or the
# cosine anneal is mis-scaled and most of the gain never arrives. save_freq is
# parked above the step count so no per-step checkpoints are written.
OVERRIDES    = ["board=False", "epochs=30", "steps_per_epoch=200",
                "opt.decay_steps=6000", "save_freq=100000",
                "checkpoint_path=./checkpoints/best_best_offline.pth"]
OUT_JSON     = "./test/bootstrap_fixed_results.json"
LOG_DIR      = "./run_logs/bootstrap"
ENTRY        = "src/finetuning.py"
CONFIG_NAME  = "CGYRO"
# ==================

LOSS_RE = re.compile(r"^FULL TEST LOSS:\s*([-+0-9.eE]+)\s*$", re.M)

_lock = __import__("threading").Lock()


def _append(record):
    """Append one result, rewriting the whole file (tiny) under a lock."""
    with _lock:
        existing = []
        if os.path.exists(OUT_JSON):
            try:
                existing = json.load(open(OUT_JSON))
            except Exception:
                existing = []
        existing.append(record)
        os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
        tmp = OUT_JSON + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=1)
        os.replace(tmp, OUT_JSON)
        return len(existing)


def run_one(split_id, gpu):
    split_dir = f"{SPLITS_ROOT}/split_{split_id}"
    manifest_path = os.path.join(split_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} missing -- generate the splits first "
            f"(python test/make_bootstrap_splits.py)")
    manifest = json.load(open(manifest_path))
    assert manifest["layout"] == "fixed", (
        f"split_{split_id} is layout={manifest['layout']!r}; Phase 1 needs 'fixed' "
        f"(re-materialize with BOOT_LAYOUT=fixed)")

    cmd = [PYTHON, ENTRY, f"--config-name={CONFIG_NAME}",
           f"dataset.dataset_root={split_dir}", *OVERRIDES]
    # LD_LIBRARY_PATH is mandatory: finetuning.py imports `bal`, which pulls in
    # matplotlib, which fails with `CXXABI_1.3.15 not found` against the system
    # libstdc++. See CLAUDE.md "Environment / running".
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": str(gpu),
           "LD_LIBRARY_PATH": CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")}

    t0 = datetime.now(timezone.utc)
    print(f"[split {split_id:2d}] gpu={gpu} start  {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"split_{split_id}.log")
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n===== STDERR =====\n")
            f.write(proc.stderr)

    hits = LOSS_RE.findall(proc.stdout)
    if not hits:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
        raise RuntimeError(
            f"split {split_id}: no 'FULL TEST LOSS:' in stdout (rc={proc.returncode}).\n"
            f"That means src/finetuning.py is missing the full-test-set eval added "
            f"at the end of run_train -- the loop's `test_loss` is a single batch and "
            f"must not be substituted. Log: {log_path}\n--- tail ---\n{tail}")
    loss = float(hits[-1])
    assert proc.returncode == 0, f"split {split_id}: rc={proc.returncode}, see {log_path}"

    record = dict(
        split_id=split_id, repeat=manifest["repeat"], fold=manifest["fold"],
        full_test_loss=loss, returncode=proc.returncode, seconds=dt, gpu=gpu,
        log=log_path, cmd=cmd,
        test_rows=manifest["test"]["rows"],
        test_valid_ky=manifest["test"]["valid_ky"],
        train_rows=manifest["train"]["rows"],
        train_groups=manifest["train"]["groups"],
        train_valid_ky=manifest["train"]["valid_ky"],
        leaked_test_rows=manifest["leaked_test_rows"],
        source_fingerprint=manifest["source_fingerprint"],
        split_fingerprint=manifest["split_fingerprint"],
        manifest_git_commit=manifest["git_commit"],
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    n = _append(record)
    print(f"[split {split_id:2d}] gpu={gpu} done   loss={loss:.6f}  "
          f"{dt / 60:.1f} min  ({n} results in {OUT_JSON})", flush=True)
    return record


def main():
    jobs = [(sid, GPUS[i % len(GPUS)]) for i, sid in enumerate(SPLIT_IDS)]
    print(f"{len(jobs)} splits, {MAX_PARALLEL} in parallel over GPUs {GPUS}")
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futs = {pool.submit(run_one, sid, gpu): sid for sid, gpu in jobs}
        for fut in futs:
            try:
                fut.result()
            except Exception as e:      # keep going; a dead split is recorded, not fatal
                errors.append((futs[fut], repr(e)))
                print(f"!! split {futs[fut]} FAILED: {e}", flush=True)

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else []
    losses = [r["full_test_loss"] for r in results if r["split_id"] in set(SPLIT_IDS)]
    print(f"\n{len(losses)}/{len(SPLIT_IDS)} splits completed -> {OUT_JSON}")
    if losses:
        import statistics as st
        # Repeat-level SE (7 df at R=8) is the honest one: folds inside a repeat
        # share training data, so sd(all 40)/sqrt(40) is anti-conservative
        # (Bengio & Grandvalet 2004). Both are printed, labelled.
        by_rep = {}
        for r in results:
            by_rep.setdefault(r["repeat"], []).append(r["full_test_loss"])
        rep_means = [st.mean(v) for v in by_rep.values()]
        print(f"  grand mean            {st.mean(losses):.6f}")
        print(f"  sd across splits      {st.pstdev(losses):.6f}   <- sigma_split")
        if len(rep_means) > 1:
            print(f"  SE (repeat-level, {len(rep_means) - 1} df) "
                  f"{st.stdev(rep_means) / len(rep_means) ** 0.5:.6f}   <- report this")
        if len(losses) > 1:
            print(f"  SE (naive, all splits)  "
                  f"{st.stdev(losses) / len(losses) ** 0.5:.6f}   <- anti-conservative")
    if errors:
        print("\nfailures:")
        for sid, e in errors:
            print(f"  split {sid}: {e}")


if __name__ == "__main__":
    main()
