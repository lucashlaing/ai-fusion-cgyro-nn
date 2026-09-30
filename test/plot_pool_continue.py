"""Regenerate acq_comparison_pool_continue.png = the offline pool-sampling +
continuous-retrain acquisition sweep, ALL 20 runs (original 10 + the 10 combos).
Pulls history straight from wandb (no CSV export). One-off: config is constants.

Solid line = original 10-fn sweep; dashed = the 10 grid-completion combos.
Labels are the canonical Separate-Budget-Select names from ACQ_GLOSSARY.md.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

API_KEY = os.environ.get("WANDB_API_KEY")  # None -> wandb falls back to ~/.netrc
ENTITY = "mhlaing-university-of-california-san-diego"
PROJECT = "CGYRO-train-fixed-op"
OUT = os.path.join(os.path.dirname(__file__), "acq_comparison_pool_continue.png")
HLINE = 0.009    # full-data offline baseline
XCUT = 300000    # drop all data points past this (curves converged well before)
XVIEW = 345000   # axis right edge -- a little space past XCUT so it's not cramped
CACHE = os.path.join(os.path.dirname(__file__), "acq_pool_continue_data.json")
# Selection must be pinned to the RECIPE, not to recency. This project also holds
# later runs that share an acquisition key but belong to other experiments --
# notably a failed 2026-08-08 batch at n_samples=50000, which "newest wins" chose
# for `random` and which has no history at all, silently dropping the baseline arm
# from the figure. Requiring the sweep's own candidate-pool size and a non-failed
# state pins every arm to the 2026-07 sweep.
N_SAMPLES = 400000
BAD_STATES = {"failed"}

# acq_func -> canonical Separate-Budget-Select label (ACQ_GLOSSARY master lookup)
LABELS = {
    "random":         "UniBin-Uniform-Random",
    "eig":            "UniBin-Uniform-EIG",
    "eig_stratified": "Uniform-Ranked-EIG",
    "res_uni_pflip":  "Deviation-Uniform-P-Flip",
    "res_uni_ran":    "Deviation-Uniform-Random",
    "strat_uni_ran":  "Uniform-Uniform-Random",
    "strat_uni_pflip":"Uniform-Uniform-P-Flip",
    "strat_eig_pflip":"Uniform-EIG-P-Flip",
    "glo_uni_ow":     "UniBin-Uniform-Importance",
    "strat_rank_ow":  "Uniform-Ranked-Importance",
    # --- 10 grid-completion combos (plotted dashed) ---
    "glo_uni_pflip":  "UniBin-Uniform-P-Flip",
    "strat_uni_ow":   "Uniform-Uniform-Importance",
    "strat_uni_eig":  "Uniform-Uniform-EIG",
    "strat_rank_ran": "Uniform-EIG-Random",
    "res_uni_ow":     "Deviation-Uniform-Importance",
    "res_uni_eig":    "Deviation-Uniform-EIG",
    "res_rank_ran":   "Deviation-EIG-Random",
    "res_rank_pflip": "Deviation-EIG-P-Flip",
    "res_rank_ow":    "Deviation-Ranked-Importance",
    "res_rank_eig":   "Deviation-EIG-EIG",
}
NEW_COMBOS = {"glo_uni_pflip", "strat_uni_ow", "strat_uni_eig", "strat_rank_ran",
              "res_uni_ow", "res_uni_eig", "res_rank_ran", "res_rank_pflip",
              "res_rank_ow", "res_rank_eig"}

import json

api = wandb.Api(api_key=API_KEY)
runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=200)

# collect offline + continuous_retrain=true runs, newest wins per acq_func
selected = {}
for r in runs:
    c = r.config
    bal = c.get("bal", {}) if isinstance(c.get("bal"), dict) else {}
    sm = bal.get("sampling_mode", c.get("bal/sampling_mode"))
    cr = bal.get("continuous_retrain", c.get("bal/continuous_retrain"))
    acq = bal.get("acquisition_function", c.get("bal/acquisition_function"))
    ns = bal.get("n_samples", c.get("bal/n_samples"))
    if (sm == "offline" and cr is True and ns == N_SAMPLES
            and r.state not in BAD_STATES and acq in LABELS and acq not in selected):
        selected[acq] = r
print(f"matched {len(selected)}/{len(LABELS)} runs")
missing = sorted(set(LABELS) - set(selected))
if missing:
    print(f"  WARNING: no run matched for {missing}")
for a in sorted(selected):
    print(f"  {a:<18}{selected[a].name:<40}{selected[a].state}")

def history(run):
    """(samples the model was trained on, loss) per BAL iteration.

    `BAL/test_loss` and `BAL/total_samples` are logged under the SAME
    `BAL/iteration` but do not describe the same model. In `src/bal_finetune.py`
    the loss is computed (:309) and logged (:323) BEFORE that round's
    acquisition, then `total_num_samples += num_acq` (:368) and
    `BAL/total_samples` is logged (:371) under the same iteration index. So

        loss[i]  is the model trained on  total_samples[i-1]

    and joining the two on the iteration index overstates every budget by one
    round -- 10,000 samples in this sweep. Iteration 0 is the pretrained model on
    the initial train set, which is never logged as a total; recover it as
    `total_samples[0] - num_samples[0]`.

    This is subtle enough to have been wrong in both directions here: the
    original code tail-aligned the two lists positionally, which landed on the
    right pairing by accident with a wrong explanation, and a 2026-09-21 "fix"
    to an iteration-index join made the numbers wrong for the first time. The
    pairing is now explicit. See PROJECT_KNOWLEDGE.md 12.

    scan_history returns only rows where EVERY requested key is present, so the
    metrics cannot be requested in one call.
    """
    loss = {r["BAL/iteration"]: r["BAL/test_loss"] for r in
            run.scan_history(keys=["BAL/iteration", "BAL/test_loss"])}
    nsamp = {r["BAL/iteration"]: r["BAL/total_samples"] for r in
             run.scan_history(keys=["BAL/iteration", "BAL/total_samples"])}
    nacq = {r["BAL/iteration"]: r["BAL/num_samples"] for r in
            run.scan_history(keys=["BAL/iteration", "BAL/num_samples"])}
    if not loss or not nsamp:
        return np.array([]), np.array([])
    initial = nsamp[0] - nacq[0]
    xs, ys = [], []
    for i in sorted(loss):
        n = initial if i == 0 else nsamp.get(i - 1)
        if n is not None:
            xs.append(n); ys.append(loss[i])
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    o = np.argsort(xs)
    return xs[o], ys[o]


# fetch histories once, truncate at the cutoff (label/sort use the 300k loss,
# not the 620k endpoint, so the legend matches what's actually drawn).
# Cached: 20 runs x 3 scan_history calls runs ~10 min.
if os.path.exists(CACHE):
    print(f"using cache {CACHE} (delete it to re-pull)")
    raw = {k: (np.array(v[0], float), np.array(v[1], float))
           for k, v in json.load(open(CACHE)).items()}
else:
    raw = {}
    for acq in selected:
        raw[acq] = history(selected[acq])
        print(f"  pulled {acq}")
    json.dump({k: [v[0].tolist(), v[1].tolist()] for k, v in raw.items()},
              open(CACHE, "w"))
    print(f"cached -> {CACHE}")

cut = {}
for acq in selected:
    x, y = raw[acq]
    if x.size:
        keep = x <= XCUT
        cut[acq] = (x[keep], y[keep])
    else:
        print(f"  {acq}: no history, skipped")

# stable plot order: originals first (solid), combos after (dashed), each by loss
order = sorted(cut.keys(), key=lambda a: (a in NEW_COMBOS, cut[a][1][-1]))
cmap = plt.get_cmap("tab20")

plt.figure(figsize=(14, 8.5))
for i, acq in enumerate(order):
    x, y = cut[acq]
    new = acq in NEW_COMBOS
    lbl = f"{LABELS[acq]}{' *' if new else ''}  ({y[-1]:.5f})"
    plt.plot(x, y, label=lbl, color=cmap(i % 20), marker="o", markersize=3,
             linewidth=1.8, linestyle="--" if new else "-", alpha=0.9)

plt.axhline(y=HLINE, color="k", ls=":", alpha=0.7, label=f"offline baseline ({HLINE})")
plt.xlim(0, XVIEW)
plt.xlabel("Total Cumulative Samples Acquired")
plt.ylabel("Test Loss")
plt.title("Pool-Sampling + Continuous-Retrain Acquisition Sweep (all 20 combos)\n"
          "solid = original 10-fn sweep, dashed* = grid-completion combos "
          "(legend = test loss @ 300k samples)")
plt.grid(True, which="both", ls="-", alpha=0.35)
plt.legend(fontsize=7.5, ncol=2, loc="upper right")
plt.tight_layout()
plt.savefig(OUT, dpi=300)
print(f"saved {OUT}")
