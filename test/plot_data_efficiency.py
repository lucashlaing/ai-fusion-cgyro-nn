"""Paired data-efficiency of each acquisition function against `random`.

This is the endpoint active learning actually claims -- "reach the same quality
with less data" -- and it is the one on which the CGYRO study has a result.
The two endpoints already in `test/plot_bootstrap.py` do not say this directly:

* **AUALC** integrates the whole curve, so a mid-curve win and a cheaper budget
  are mixed into one number.
* The **final round** is degenerate here. By round 30 every arm holds ~1626-1684
  of the pool's ~1725 unique physics hashes (PROJECT_KNOWLEDGE.md 13.3), so
  selection has nothing left to decide and the arms re-converge by construction.
  `random` wins that endpoint on 34/35 contrasts, which says the budget ran past
  the point of interest -- not that acquisition does nothing.

**Estimator.** For each split s independently:

    target(s)       = random's loss at its FINAL round on split s
    n_a(s)          = the first cumulative n at which arm a's best-so-far curve
                      reaches target(s)
    n_random(s)     = the same rule applied to `random` itself
    saving_a(s)     = 1 - n_a(s) / n_random(s)

Both arms are measured by the identical rule, which is the whole point:
`random`'s own curve dips below its final value well before its final round
(mean n ~1350 against a final n of ~1681), so comparing an arm's crossing point
against random's *final* n would manufacture a ~17% saving out of curve noise
alone. Against random's own crossing point that bias cancels.

Best-so-far (running minimum) rather than the raw curve, because a budget once
sufficient stays sufficient; the raw curve re-crosses the target several times
on a noisy split and "first crossing" would then be an artefact of where the
noise happened to land.

**Pairing is everything.** sigma_split is 0.3562 across splits against a 36-way
effect span of 0.127 (15.15), so the unpaired difference of two means is
meaningless here. Everything below is a within-split contrast, and the Wilcoxon
is paired. An arm that never reaches the target on split s drops that split for
that arm only (pairwise complete-case, count reported) -- never imputed, because
failing to reach the target is itself informative and imputing it would flatter
the arm.

Outputs `cgyro/cgyro_data_efficiency.{png,md}`. House style follows
`test/plot_pool_continue.py`: dpi 300, grid alpha 0.35, tab20.
"""
import os
import json

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === USER INPUT ===
HERE      = os.path.dirname(os.path.abspath(__file__))
IN_JSON   = os.path.join(HERE, "bootstrap_data.json")
OUTDIR    = os.path.join(HERE, "cgyro")
OUT_PNG   = os.path.join(OUTDIR, "cgyro_data_efficiency.png")
OUT_MD    = os.path.join(OUTDIR, "cgyro_data_efficiency.md")
BASELINE  = "random"
MIN_PAIRS = 8      # splits on which BOTH arms must reach the target
TOP_N     = 14     # arms drawn in the figure (the table carries all of them)
FIGSIZE   = (13, 9)
# (Separate, Budget, Select) -- verbatim from ACQ_GLOSSARY.md. NOT derivable
# from the config key: `res_rank_ran` is Deviation-EIG-Random while
# `res_rank_ow` is Deviation-Ranked-Importance.
SPEC = {
    "random":          ("UniBin",    "Uniform", "Random"),
    "eig":             ("UniBin",    "Uniform", "EIG"),
    "glo_uni_ow":      ("UniBin",    "Uniform", "Importance"),
    "glo_uni_pflip":   ("UniBin",    "Uniform", "P-Flip"),
    "glo_uni_lcmd":    ("UniBin",    "Uniform", "LCMD"),
    "direct":          ("UniBin",    "Uniform", "DIRECT"),
    "strat_uni_ran":   ("Uniform",   "Uniform", "Random"),
    "strat_uni_pflip": ("Uniform",   "Uniform", "P-Flip"),
    "strat_uni_ow":    ("Uniform",   "Uniform", "Importance"),
    "strat_uni_lcmd":  ("Uniform",   "Uniform", "LCMD"),
    "strat_uni_eig":   ("Uniform",   "Uniform", "EIG"),
    "strat_rank_ran":  ("Uniform",   "EIG",     "Random"),
    "strat_rank_ow":   ("Uniform",   "Ranked",  "Importance"),
    "strat_rank_lcmd": ("Uniform",   "Ranked",  "LCMD"),
    "strat_eig_pflip": ("Uniform",   "EIG",     "P-Flip"),
    "eig_stratified":  ("Uniform",   "Ranked",  "EIG"),
    "res_uni_ran":     ("Deviation", "Uniform", "Random"),
    "res_uni_pflip":   ("Deviation", "Uniform", "P-Flip"),
    "res_uni_ow":      ("Deviation", "Uniform", "Importance"),
    "res_uni_eig":     ("Deviation", "Uniform", "EIG"),
    "res_uni_lcmd":    ("Deviation", "Uniform", "LCMD"),
    "res_rank_ran":    ("Deviation", "EIG",     "Random"),
    "res_rank_pflip":  ("Deviation", "EIG",     "P-Flip"),
    "res_rank_ow":     ("Deviation", "Ranked",  "Importance"),
    "res_rank_lcmd":   ("Deviation", "Ranked",  "LCMD"),
    "res_rank_eig":    ("Deviation", "EIG",     "EIG"),
    "rho_uni_ran":     ("Rho",       "Uniform", "Random"),
    "rho_uni_pflip":   ("Rho",       "Uniform", "P-Flip"),
    "rho_uni_ow":      ("Rho",       "Uniform", "Importance"),
    "rho_uni_eig":     ("Rho",       "Uniform", "EIG"),
    "rho_uni_lcmd":    ("Rho",       "Uniform", "LCMD"),
    "rho_rank_ran":    ("Rho",       "EIG",     "Random"),
    "rho_rank_ow":     ("Rho",       "Ranked",  "Importance"),
    "rho_rank_lcmd":   ("Rho",       "EIG",     "LCMD"),
    "rho_rank_pflip":  ("Rho",       "EIG",     "P-Flip"),
    "rho_rank_eig":    ("Rho",       "EIG",     "EIG"),
}
LABEL = {k: "-".join(v) for k, v in SPEC.items()}
# ==================


def load():
    d = json.load(open(IN_JSON))
    usable = d["complete_case_splits"]
    cell = {(r["split_id"], r["acq"]): r for r in d["runs"]}
    acqs = [a for a in d["expect_acqs"] if a in SPEC]
    return usable, cell, acqs


def curve(cell, s, a):
    """(n, loss) for one cell, ordered by ITERATION -- not by n.

    Sorting by n would silently reorder the two or three rounds where dedup
    makes the cumulative count tie, and the running minimum below is
    order-dependent.
    """
    pts = sorted(cell[(s, a)]["points"], key=lambda p: p["iteration"])
    return (np.array([p["n"] for p in pts], dtype=float),
            np.array([p["loss"] for p in pts], dtype=float))


def budget_to_target(n, loss, target):
    """First cumulative n whose best-so-far loss reaches `target`; NaN if never."""
    hit = np.where(np.minimum.accumulate(loss) <= target)[0]
    return n[hit[0]] if len(hit) else np.nan


def efficiency(cell, usable, a):
    """Paired budget-to-target for arm `a` against the baseline, over splits."""
    na, nb = [], []
    for s in usable:
        xb, yb = curve(cell, s, BASELINE)
        xa, ya = curve(cell, s, a)
        target = yb[-1]
        va = budget_to_target(xa, ya, target)
        vb = budget_to_target(xb, yb, target)
        if np.isfinite(va) and np.isfinite(vb):
            na.append(va)
            nb.append(vb)
    return np.array(na), np.array(nb)


def summarize(cell, usable, acqs):
    """Paired rows for every arm clearing MIN_PAIRS, plus the ones that do not.

    The second return value is NOT bookkeeping. An arm drops out here because it
    failed to reach `random`'s final quality on enough splits -- i.e. because it
    is WORSE -- so the surviving table is a favourably-selected subset and a
    mean over it would overstate the field. Report both.
    """
    rows, dropped = [], []
    for a in acqs:
        if a == BASELINE:
            continue
        na, nb = efficiency(cell, usable, a)
        if len(na) < MIN_PAIRS:
            dropped.append((a, LABEL[a], len(na)))
            continue
        # Mean of the per-split savings, not 1 - mean(na)/mean(nb): the ratio of
        # means is not the mean of ratios, and only the latter is the paired
        # quantity the Wilcoxon below is testing.
        sav = 1.0 - na / nb
        try:
            p = float(stats.wilcoxon(na, nb).pvalue)
        except ValueError:          # all differences zero
            p = 1.0
        rows.append({
            "acq": a, "label": LABEL[a],
            "n_arm": float(na.mean()), "n_rand": float(nb.mean()),
            "saving": float(sav.mean()), "sav_split": sav,
            "wins": int((na < nb).sum()), "S": len(na), "p": p,
        })
    rows.sort(key=lambda r: -r["saving"])
    dropped.sort(key=lambda t: t[2])
    return rows, dropped


def write_md(rows, dropped, usable):
    lines = [
        "# CGYRO BAL — paired data efficiency vs `random`",
        "",
        f"- splits (complete case): **{len(usable)}** -> {usable}",
        "- **target** = `random`'s final-round loss, recomputed per split",
        "- **budget** = first cumulative n at which the best-so-far curve reaches that "
        "target, applied by the SAME rule to the arm and to `random` itself",
        "- saving = mean over splits of `1 - n_arm/n_random` (mean of ratios, the paired "
        "quantity; not the ratio of means)",
        "- `wins` counts splits where the arm reached the target on strictly less data; "
        "`p` is a paired two-sided Wilcoxon on the budgets",
        "- splits where either arm never reaches the target are dropped **for that arm "
        "only** and shown in `S`; never imputed",
        "",
        "> Scale check before reading any row: a single arm's split-to-split sd of final "
        "loss is ~0.32 (PROJECT_KNOWLEDGE.md 15.15). These are paired within-split "
        "contrasts precisely because the unpaired comparison is hopeless.",
        "",
        "> `p` is unadjusted. With 35 contrasts, Holm at m=35 cannot clear 0.05 from an "
        "S=10 exact Wilcoxon at all (floor 2/2^10 x 35 = 0.0684, 15.17), so these are "
        "reported as effect sizes with their nominal p, not as a multiplicity-controlled "
        "screen. The pre-specified reading is the Importance (output-weighted) family.",
        "",
        f"> **Selection caveat.** {len(rows)} of {len(rows) + len(dropped)} contrasts reach "
        f"the target on at least {MIN_PAIRS} splits and appear below; the other "
        f"{len(dropped)} are listed after the table. They drop out *because they are "
        "worse* -- they never got to random's final quality on enough splits -- so this "
        "table is a favourably-selected subset. Do not average it and call it the field.",
        "",
        "| strategy | Separate-Budget-Select | n to target | random's n | data saved | wins/S | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        star = " **" if r["p"] < 0.05 else ""
        end = "**" if r["p"] < 0.05 else ""
        lines.append(
            f"| `{r['acq']}` | {r['label']} | {r['n_arm']:.0f} | {r['n_rand']:.0f} | "
            f"{star}{r['saving']:+.1%}{end} | {r['wins']}/{r['S']} | {r['p']:.4f} |")
    if dropped:
        lines += [
            "",
            f"## Reached the target on fewer than {MIN_PAIRS} splits (excluded above)",
            "",
            "Excluded for insufficient pairs, not for their effect size. Reaching "
            "`random`'s final quality on few splits is a poor result, so read these as "
            "the bottom of the ranking rather than as missing data.",
            "",
            "| strategy | Separate-Budget-Select | splits reaching target |",
            "|---|---|---|",
        ]
        lines += [f"| `{a}` | {lab} | {n}/{len(usable)} |" for a, lab, n in dropped]
    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print("saved", OUT_MD)


def plot(rows):
    top = rows[:TOP_N]
    y = np.arange(len(top))[::-1]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    wins = []

    for i, (r, yi) in enumerate(zip(top, y)):
        sig = r["p"] < 0.05
        ax.barh(yi, r["saving"], height=0.62, color=cmap(0 if sig else 1),
                alpha=0.95 if sig else 0.55, zorder=2,
                edgecolor="#0b0b0b" if sig else "none", linewidth=1.1)
        # Per-split points: the bar is a mean over <=10 splits and the spread
        # matters more than the mean does. Jittered deterministically.
        jit = np.linspace(-0.17, 0.17, len(r["sav_split"]))
        ax.scatter(r["sav_split"], np.full(len(jit), yi) + jit, s=11,
                   color="#0b0b0b", alpha=0.45, zorder=3, linewidths=0)
        wins.append(f"{r['wins']}/{r['S']}")

    ax.axvline(0, color="#0b0b0b", lw=1.2, zorder=4)
    # wins/S on a twinned right-hand axis: inside the data area it landed on the
    # per-split dots, which are the point of the figure.
    rax = ax.twinx()
    rax.set_ylim(ax.get_ylim())
    rax.set_yticks(y)
    rax.set_yticklabels(wins, fontsize=8.5, color="#52514e")
    rax.set_ylabel("splits won", fontsize=9, color="#52514e")
    rax.tick_params(length=0)
    for sp in rax.spines.values():
        sp.set_visible(False)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']}  (p={r['p']:.3f})" for r in top], fontsize=9)
    ax.set_xlabel("Data saved vs `random` to reach the same test loss\n"
                  "(mean over splits of 1 - n_arm/n_random; dots = individual splits)")
    ax.grid(True, axis="x", ls="-", alpha=0.35, zorder=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(
        "CGYRO BAL: budget needed to reach `random`'s final quality\n"
        f"paired within split, 10 leak-free splits, 30 rounds x 50 samples; "
        f"top {len(top)} of {len(rows)} arms reaching the target\n"
        "filled = nominal p < 0.05 (unadjusted; see the .md on multiplicity)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print("saved", OUT_PNG)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    usable, cell, acqs = load()
    rows, dropped = summarize(cell, usable, acqs)
    print(f"{len(rows)} arms with >= {MIN_PAIRS} paired splits, of {len(acqs) - 1} contrasts")
    print(f"{len(dropped)} excluded for too few splits reaching the target "
          f"(they are worse, not missing)\n")
    print(f"{'arm':<18}{'n arm':>8}{'n rand':>8}{'saved':>9}{'wins':>8}{'p':>10}")
    for r in rows:
        print(f"{r['acq']:<18}{r['n_arm']:>8.0f}{r['n_rand']:>8.0f}"
              f"{r['saving']:>8.1%}{r['wins']:>5}/{r['S']:<2}{r['p']:>10.4f}")
    if dropped:
        print("\nexcluded (splits reaching target):")
        for a, lab, n in dropped:
            print(f"  {a:<18}{lab:<28}{n}/{len(usable)}")
    write_md(rows, dropped, usable)
    plot(rows)


if __name__ == "__main__":
    main()
