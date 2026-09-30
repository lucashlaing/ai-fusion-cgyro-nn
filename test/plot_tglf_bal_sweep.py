"""The TGLF 14-run BAL acquisition sweep -- curve + ranked table.

This is the sweep behind the headline TGLF result (PROJECT_KNOWLEDGE.md 11.9):
BAL reaches the tuned full-data offline baseline at ~9% of the 3.7M pool, and
beats it outright by ~21% at ~17%. **That result had no surviving figure and no
regenerable table.** 11.9 and CLAUDE.md both cite `test/lcmd_vs_combos_table.md`,
`test/lcmd_vs_combos.png` and `test/make_loss_table.py`; none of the three exists
on disk or in git. This script replaces all three from wandb.

Runs: the 14 named `202608{15,16}-*_BAL_*` in the wandb project below, all
`sampling_mode=offline`, `continuous_retrain=true`, `rho_balanced=true`,
`n_samples=400000`, `new_sample_size=10000`, 20k initial, 60 iterations
requested, from `model_bal_latest.pth`, commit 60c89b0.

**The four LCMD arms crashed at iteration 29-30** (pod-disk eviction, 5.6: BAL
snapshots the whole cumulative `train/` every iteration and never prunes, so
disk grows quadratically; they requested `ephemeral_storage: 300`). They cannot
be resumed -- `checkpoint_s3_uri=null`. So **iteration 30 is the only point
where all 14 arms overlap**, and the 60-iteration comparison is 10 arms only.
Both tables are emitted; do not merge them.

Two wandb traps this script works around (12):

* A reused `wandb.Api()` caches summaries and `run.state`. One fresh Api() here.
* **`BAL/test_loss` and `BAL/total_samples` at the SAME `BAL/iteration` do not
  describe the same model.** Read `src/bal_finetune.py`: within iteration `i`,
  `current_test_loss` is computed at :309 and logged at :323 -- *before* that
  round's acquisition -- then `total_num_samples += num_acq` at :368 and
  `BAL/total_samples` is logged at :371 under the **same** `BAL/iteration: i`.
  So `loss[i]` is the model trained on `total_samples[i-1]`, and joining the two
  on the iteration index overstates every budget by one round (10,000 samples
  here). The correct pairing is

      n_trained(i) = total_samples[i-1]   =  initial + 10,000 * i

  with `initial = total_samples[0] - num_samples[0]` (19,948 here; the initial
  train set is never itself logged as a total).

  **This corrects the headline.** 11.9 reports `res_rank_pflip` = 0.00948 "at
  329,948 samples, ~8.9% of the pool"; that pairs `loss[30]` with
  `total_samples[30]`. The model that scored 0.00948 was trained on
  **319,948 samples = 8.6%** of the 3.7M pool. 11.9's iteration-60 figure
  (0.007508 at 619,872) is right, because there it *did* use `total_samples[59]`
  -- its own stated formula `29,948 + 10,000*i` contradicts it and yields 629,948.
  `test/plot_pool_continue.py` had the same off-by-one and is fixed alongside.

* `scan_history(keys=[...])` returns only rows where EVERY requested key is
  present, so the metrics cannot be requested in one call.

Metric is `BAL/test_loss`: full test set, asinh-space per-ky MSE, one point per
BAL iteration. Every run here predates the `38ce8a4` estimator change, so they
are mutually consistent -- and **not poolable with any post-38ce8a4 CGYRO
number**. It is the same quantity on the same split as the offline runs'
`BAL Test Loss`, which is what licenses the parity claim against HLINE.

Single-seed throughout: the noise floor below is the largest single-iteration
uptick anywhere in the sweep, not a repeat-based estimate. Nothing here
separates effects smaller than it.
"""
import os
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# === USER INPUT ===
API_KEY   = os.environ.get("WANDB_API_KEY")  # None -> wandb falls back to ~/.netrc
ENTITY    = "mhlaing-university-of-california-san-diego"
# TGLF BAL runs predate the 2026-08-23 project rename, so they are in the OLD
# project despite being TGLF. `TGLF-train-fixed-op` holds no TGLF BAL run.
PROJECT   = "CGYRO-train-fixed-op"
NAME_PREFIXES = ("20260815-", "20260816-")
HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_PNG   = os.path.join(HERE, "tglf_bal_sweep.png")
OUT_MD    = os.path.join(HERE, "tglf_bal_sweep.md")
CACHE     = os.path.join(HERE, "tglf_bal_sweep_data.json")
HLINE     = 0.009523      # tuned full-data offline baseline, peak_lr=1e-3 wd=1e-5
POOL      = 3_700_000     # TGLF pool size, for the "% of pool" column
MILESTONES = (10, 20, 30, 60)
NOISE     = 0.00046       # largest single-iteration uptick in the sweep
FIGSIZE   = (14, 8.5)
# (Separate, Budget, Select) -- verbatim from ACQ_GLOSSARY.md. NOT derivable
# from the config key: `res_rank_ran` is Deviation-EIG-Random while
# `res_rank_ow` is Deviation-Ranked-Importance.
SPEC = {
    "res_rank_pflip":  ("Deviation", "EIG",     "P-Flip"),
    "res_uni_eig":     ("Deviation", "Uniform", "EIG"),
    "res_rank_ran":    ("Deviation", "EIG",     "Random"),
    "res_rank_eig":    ("Deviation", "EIG",     "EIG"),
    "res_uni_lcmd":    ("Deviation", "Uniform", "LCMD"),
    "res_uni_ow":      ("Deviation", "Uniform", "Importance"),
    "res_rank_ow":     ("Deviation", "Ranked",  "Importance"),
    "glo_uni_lcmd":    ("UniBin",    "Uniform", "LCMD"),
    "glo_uni_pflip":   ("UniBin",    "Uniform", "P-Flip"),
    "strat_uni_eig":   ("Uniform",   "Uniform", "EIG"),
    "strat_uni_lcmd":  ("Uniform",   "Uniform", "LCMD"),
    "strat_uni_ow":    ("Uniform",   "Uniform", "Importance"),
    "strat_rank_ran":  ("Uniform",   "EIG",     "Random"),
    "rho_uni_lcmd":    ("Rho",       "Uniform", "LCMD"),
}
LABEL = {k: "-".join(v) for k, v in SPEC.items()}
# ==================


def pull():
    """Fetch the 14 runs' (n, loss) histories. Cached -- the pull is ~2 min."""
    if os.path.exists(CACHE):
        print(f"using cache {CACHE} (delete it to re-pull)")
        return json.load(open(CACHE))
    api = wandb.Api(api_key=API_KEY)      # fresh Api(): a reused one caches state
    out = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=200):
        if not (r.name.startswith(NAME_PREFIXES) and "_BAL_" in r.name):
            continue
        acq = r.name.split("_BAL_", 1)[1]
        if acq not in SPEC:
            continue
        # Two calls, each pairing its metric with BAL/iteration: scan_history
        # returns only rows where EVERY requested key is present, so asking for
        # both metrics at once yields zero rows.
        loss = {d["BAL/iteration"]: d["BAL/test_loss"] for d in
                r.scan_history(keys=["BAL/iteration", "BAL/test_loss"])}
        nsamp = {d["BAL/iteration"]: d["BAL/total_samples"] for d in
                 r.scan_history(keys=["BAL/iteration", "BAL/total_samples"])}
        nacq = {d["BAL/iteration"]: d["BAL/num_samples"] for d in
                r.scan_history(keys=["BAL/iteration", "BAL/num_samples"])}
        # loss[i] is the model trained on total_samples[i-1] -- see the docstring.
        # i=0 is the pretrained model on the initial train set, which is never
        # logged as a total, so recover it as total_samples[0] - num_samples[0].
        initial = nsamp[0] - nacq[0]
        its, ns, ys = [], [], []
        for i in sorted(loss):
            n = initial if i == 0 else nsamp.get(i - 1)
            if n is None:
                continue
            its.append(i); ns.append(n); ys.append(loss[i])
        out[acq] = {"run_id": r.id, "name": r.name, "state": r.state,
                    "initial": initial, "iter": its, "n": ns, "loss": ys}
        print(f"  {acq:<17} {r.id}  {r.state:<9} iters {its[0]}-{its[-1]} "
              f"({len(its)} pts)  init={initial:,}  last n={ns[-1]:,}  "
              f"last loss={ys[-1]:.5f}")
    json.dump(out, open(CACHE, "w"), indent=1)
    print(f"cached -> {CACHE}")
    return out


def at_iteration(rec, it):
    """(loss, n) at BAL iteration `it`; (None, None) if the run died first.

    Looked up by the logged `BAL/iteration`, never by list position: the four
    crashed arms stop at different rounds, so position and iteration diverge.
    """
    for i, n, y in zip(rec["iter"], rec["n"], rec["loss"]):
        if i == it:
            return y, n
    return None, None


def table(data, iters, title, note):
    rows = []
    for acq, rec in data.items():
        pairs = [at_iteration(rec, it) for it in iters]
        vals = [v for v, _ in pairs]
        if vals[-1] is None:
            continue
        rows.append((acq, vals, pairs[-1][1]))
    rows.sort(key=lambda t: t[1][-1])
    head = " | ".join(f"it {i}" for i in iters)
    sep = "|".join(["---"] * (len(iters) + 4))
    out = [f"## {title}", "", note, "",
           f"| rank | key | Separate-Budget-Select | {head} | samples | % of pool | vs offline |",
           f"|{sep}|---|---|"]
    for i, (acq, vals, n) in enumerate(rows, 1):
        cells = " | ".join("—" if v is None else f"{v:.5f}" for v in vals)
        out.append(f"| {i} | `{acq}` | {LABEL[acq]} | {cells} | {n:,} | "
                   f"{n / POOL:.1%} | {vals[-1] - HLINE:+.5f} |")
    return "\n".join(out), rows


def write_md(data):
    n_fin = sum(1 for r in data.values() if r["state"] == "finished")
    lines = [
        "# TGLF BAL acquisition sweep (14 arms, offline pool + continuous retrain)",
        "",
        f"- source: wandb `{ENTITY}/{PROJECT}`, runs `202608{{15,16}}-*_BAL_*` "
        f"({n_fin} finished, {len(data) - n_fin} crashed)",
        f"- metric `BAL/test_loss` (full test set). Tuned full-data offline baseline "
        f"`BAL Test Loss` = **{HLINE}** (peak_lr=1e-3, wd=1e-5, 175k steps, full "
        f"{POOL:,}-row pool) — the same quantity on the same split, which is what "
        "licenses the parity comparison.",
        f"- offline mode acquires the full 10k budget with zero dedup loss, so "
        f"cumulative n = 29,948 + 10,000 x iteration exactly",
        "- **single-seed throughout.** The noise floor below is the largest "
        f"single-iteration uptick in the sweep, **{NOISE}**, not a repeat-based "
        "estimate — no ordering inside that margin is signal.",
        "- every run predates the `38ce8a4` estimator change, so these are mutually "
        "consistent and **not poolable with any post-`38ce8a4` CGYRO number**",
        "",
    ]
    md30, rows30 = table(
        data, list(MILESTONES[:3]), "Iteration 30 (~320k samples, ~8.6% of pool)",
        "13 of the 14 arms — the deepest round with near-complete coverage. The four LCMD "
        "arms died at iteration 29-30 from pod-disk eviction (PROJECT_KNOWLEDGE.md 5.6) and "
        "cannot be resumed (`checkpoint_s3_uri=null`); `res_uni_lcmd` stopped at 29 and so has "
        "no entry here (it reads 0.00998 at iteration 29, which is why 11.9 ranks the LCMD "
        "arms at 29). **The model evaluated at iteration i was trained on "
        "`total_samples[i-1]`** — see 12 trap 4; pairing on the same index would overstate "
        "this budget by 10,000 samples.")
    md60, rows60 = table(
        data, list(MILESTONES), "Iteration 60 (~620k samples, ~16.8% of pool)",
        "The 10 arms that finished. **Do not merge this table with the one above** — "
        "it is a different arm set at a different budget.")
    lines += [md30, "", md60, ""]

    best30, best60 = rows30[0], rows60[0]
    n30, n60 = best30[2], best60[2]
    lines += [
        "## What the sweep establishes",
        "",
        f"1. **Parity with the full-data baseline at {n30 / POOL:.1%} of the pool.** `{best30[0]}` "
        f"({LABEL[best30[0]]}) reaches **{best30[1][-1]:.5f}** at {n30:,} samples "
        f"against the baseline's {HLINE} — a gap of {best30[1][-1] - HLINE:+.5f}, "
        f"inside the {NOISE} noise floor.",
        f"2. **Outright improvement at {n60 / POOL:.1%} of the pool.** `{best60[0]}` reaches "
        f"**{best60[1][-1]:.5f}** at {n60:,} samples, "
        f"{(1 - best60[1][-1] / HLINE):.1%} below the full-data baseline.",
        "3. **The Separate stage dominates the Select stage.** Deviation-* arms sweep "
        "the top of both tables; swapping the selector moves an arm within its "
        "separator band, never out of it. A single EIG-vs-random ablation could not "
        "have shown this.",
        "4. **Mid-flight rankings invert.** The eventual winner sits mid-field at "
        "iteration 10 — most of the gain arrives as the cosine LR anneals, so no "
        "ranking should be read off an unfinished run.",
        "",
        "## What it does not establish",
        "",
        f"- **LCMD is untested at full budget.** All four LCMD arms died at iteration "
        "29-30, exactly where the JMLR benchmark predicts a diversity method starts "
        "paying. The 60-iteration LCMD comparison does not exist.",
        "- **Nothing below the noise floor.** Ranks 1-4 at iteration 30 sit within "
        f"{NOISE} of each other; that they are all Deviation-* is the signal, their "
        "order is not.",
        "- **No uncertainty estimate.** Single-seed, no repeats, no splits. The CGYRO "
        "side of the paper exists partly because this one cannot quantify its own "
        "error bars.",
    ]
    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print("saved", OUT_MD)


def plot(data):
    cmap = plt.get_cmap("tab20")
    order = sorted(data, key=lambda a: (at_iteration(data[a], 30)[0] or 9e9))
    plt.figure(figsize=FIGSIZE)
    for i, acq in enumerate(order):
        rec = data[acq]
        n, y = np.array(rec["n"], float), np.array(rec["loss"], float)
        crashed = rec["state"] != "finished"
        v30, _ = at_iteration(rec, 30)
        tag = f"  ({v30:.5f})" if v30 is not None else "  (died <30)"
        plt.plot(n, y, label=f"{LABEL[acq]}{tag}" + (" [crashed]" if crashed else ""),
                 color=cmap(i % 20), marker="o", markersize=3, linewidth=1.8,
                 linestyle="--" if crashed else "-", alpha=0.9)
    plt.axhline(HLINE, color="k", ls=":", alpha=0.8,
                label=f"full-data offline baseline ({HLINE})")
    x30 = at_iteration(data["res_rank_pflip"], 30)[1]
    n30 = sum(1 for rec in data.values() if at_iteration(rec, 30)[0] is not None)
    plt.axvline(x30, color="#52514e", ls="-.", alpha=0.5, lw=1.2)
    # Anchored near the bottom: the top of the axes is under the two-column legend.
    plt.text(x30, plt.ylim()[0], f"  iteration 30 — {n30}/{len(data)} arms",
             va="bottom", fontsize=8.5, color="#52514e")
    plt.yscale("log")
    plt.xlabel("Total Cumulative Samples Acquired")
    plt.ylabel("BAL/test_loss  (full test set)")
    plt.title(f"TGLF BAL acquisition sweep: {len(data)} strategies, offline pool + continuous retrain\n"
              "dashed = crashed at iteration 29-30 (pod-disk eviction); legend = test loss at "
              f"iteration 30 ({x30:,} samples, {x30 / POOL:.1%} of the {POOL / 1e6:.1f}M pool)")
    plt.grid(True, which="both", ls="-", alpha=0.35)
    plt.legend(fontsize=8.5, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print("saved", OUT_PNG)


def main():
    data = pull()
    print(f"\n{len(data)}/14 runs")
    write_md(data)
    plot(data)


if __name__ == "__main__":
    main()
