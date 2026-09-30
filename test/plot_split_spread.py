"""Per-split curves for ONE acquisition function, plus their mean.

Shows how much of what you see in an averaged curve is actually split-to-split
variation. This is the single most important plot for reading every other CGYRO
figure: the seed/split term on CGYRO is ~0.31 while the entire acquisition spread
is ~0.05, so one arm's 10 splits are spread roughly 6x wider than the gap between
any two arms' means.

House style is test/plot_pool_continue.py -> acq_comparison_pool_continue.png:
tab20 cycled, 'o' markers ms=3, lw 1.8, dotted black offline baseline, grid
alpha 0.35, dpi 300. The mean is drawn thick black on top of the thin per-split
curves. **No +/-1 sd band** -- it was removed 2026-09-16: the point of this figure
is the individual split curves, and a shaded band over ten of them hides exactly
what it is meant to show. The sd is still reported, in the title and on stdout.

Usage:
    python test/plot_split_spread.py                 # default set
    python test/plot_split_spread.py eig random      # named acqs
    python test/plot_split_spread.py --all           # every acq in the pull
"""
import os
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bootstrap_data.json")
FIXED = os.path.join(HERE, "bootstrap_fixed_results.json")
OUTDIR = os.path.join(HERE, "cgyro")
DEFAULT = ["eig", "random"]

# verbatim from ACQ_GLOSSARY.md -- the triple is NOT derivable from the key
LABEL = {
    "random": "UniBin-Uniform-Random", "eig": "UniBin-Uniform-EIG",
    "glo_uni_ow": "UniBin-Uniform-Importance", "glo_uni_pflip": "UniBin-Uniform-P-Flip",
    "glo_uni_lcmd": "UniBin-Uniform-LCMD",
    "strat_uni_ran": "Uniform-Uniform-Random", "strat_uni_pflip": "Uniform-Uniform-P-Flip",
    "strat_uni_ow": "Uniform-Uniform-Importance", "strat_uni_lcmd": "Uniform-Uniform-LCMD",
    "strat_rank_ran": "Uniform-EIG-Random", "strat_rank_ow": "Uniform-Ranked-Importance",
    "strat_rank_lcmd": "Uniform-Ranked-LCMD",
    "res_uni_ran": "Deviation-Uniform-Random", "res_uni_pflip": "Deviation-Uniform-P-Flip",
    "res_uni_ow": "Deviation-Uniform-Importance", "res_uni_eig": "Deviation-Uniform-EIG",
    "res_uni_lcmd": "Deviation-Uniform-LCMD", "res_rank_ran": "Deviation-EIG-Random",
    "res_rank_pflip": "Deviation-EIG-P-Flip", "res_rank_ow": "Deviation-Ranked-Importance",
    "res_rank_lcmd": "Deviation-Ranked-LCMD",
    "rho_uni_ran": "Rho-Uniform-Random", "rho_uni_pflip": "Rho-Uniform-P-Flip",
    "rho_uni_ow": "Rho-Uniform-Importance", "rho_uni_eig": "Rho-Uniform-EIG",
    "rho_uni_lcmd": "Rho-Uniform-LCMD", "rho_rank_ran": "Rho-EIG-Random",
    "rho_rank_ow": "Rho-Ranked-Importance",
    "rho_rank_lcmd": "Rho-EIG-LCMD", "res_rank_eig": "Deviation-EIG-EIG",
    "direct": "UniBin-Uniform-DIRECT", "rho_rank_pflip": "Rho-EIG-P-Flip",
    "strat_eig_pflip": "Uniform-EIG-P-Flip", "strat_uni_eig": "Uniform-Uniform-EIG",
    "eig_stratified": "Uniform-Ranked-EIG", "rho_rank_eig": "Rho-EIG-EIG",
}


def offline_ref(splits):
    if not os.path.exists(FIXED):
        return None
    rows = [r for r in json.load(open(FIXED)) if r["split_id"] in set(splits)]
    vals = [r["full_test_loss"] for r in rows if "full_test_loss" in r]
    return float(np.mean(vals)) if vals else None


def plot_one(acq, usable, cell, ref):
    # Aggregate on ITERATION INDEX -- the only coordinate aligned across splits,
    # since the initial train size varies with dedup -- and plot at the median n.
    iters = sorted({p["iteration"] for s in usable for p in cell[(s, acq)]["points"]})
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(14, 8.5))

    per_split = {}
    for s in usable:
        pts = sorted(cell[(s, acq)]["points"], key=lambda p: p["iteration"])
        per_split[s] = (np.array([p["n"] for p in pts], dtype=float),
                        np.array([p["loss"] for p in pts], dtype=float))

    for i, s in enumerate(sorted(per_split)):
        x, y = per_split[s]
        plt.plot(x, y, color=cmap(i % 20), marker="o", markersize=3, linewidth=1.8,
                 alpha=0.75, label=f"split {s}  ({y[-1]:.4f})")

    xs, mu = [], []
    for it in iters:
        v = [p["loss"] for s in usable for p in cell[(s, acq)]["points"]
             if p["iteration"] == it]
        n = [p["n"] for s in usable for p in cell[(s, acq)]["points"]
             if p["iteration"] == it]
        if len(v) < 2:
            continue
        xs.append(np.median(n))
        mu.append(float(np.mean(v)))
    xs, mu = np.array(xs), np.array(mu)

    plt.plot(xs, mu, color="k", linewidth=3.0, marker="o", markersize=4,
             label=f"MEAN of {len(usable)} splits  ({mu[-1]:.4f})", zorder=5)
    if ref is not None:
        plt.axhline(y=ref, color="k", ls=":", alpha=0.7,
                    label=f"offline baseline ({ref:.4f})")

    plt.xlabel("Total Cumulative Samples Acquired")
    plt.ylabel("Test Loss")
    finals = np.array([per_split[s][1][-1] for s in sorted(per_split)])
    plt.title(f"CGYRO BAL Bootstrap: per-split spread for {LABEL.get(acq, acq)} "
              f"(`{acq}`)\n"
              f"all {len(usable)} splits and their mean — at the final round: "
              f"mean {finals.mean():.4f}, sd {finals.std(ddof=1):.4f}, "
              f"splits span {finals.min():.4f}–{finals.max():.4f} "
              f"(range {finals.max() - finals.min():.4f})")
    plt.grid(True, which="both", ls="-", alpha=0.35)
    plt.legend(fontsize=8.5, ncol=2, loc="upper right")
    plt.tight_layout()
    p = os.path.join(OUTDIR, f"cgyro_split_spread_{acq}.png")
    plt.savefig(p, dpi=300)
    plt.close()
    finals = np.array([per_split[s][1][-1] for s in sorted(per_split)])
    print(f"saved {p}")
    print(f"   final round: mean {finals.mean():.4f}  sd {finals.std(ddof=1):.4f}  "
          f"min {finals.min():.4f}  max {finals.max():.4f}  "
          f"range {finals.max() - finals.min():.4f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = json.load(open(DATA))
    usable = d["complete_case_splits"]
    cell = {(r["split_id"], r["acq"]): r for r in d["runs"]}
    have = set(d["expect_acqs"])

    args = sys.argv[1:]
    acqs = sorted(have) if args[:1] == ["--all"] else (args or DEFAULT)
    bad = [a for a in acqs if a not in have]
    if bad:
        sys.exit(f"not in {DATA}: {bad}\navailable: {sorted(have)}")

    ref = offline_ref(usable)
    for a in acqs:
        plot_one(a, usable, cell, ref)


if __name__ == "__main__":
    main()
