"""Separator- and selector-focused views of the CGYRO bootstrap study.

House style is test/plot_pool_continue.py -> acq_comparison_pool_continue.png:
tab20 cycled, 'o' markers ms=3, lw 1.8, curves ordered by final loss with that
value in the legend, dotted black offline baseline, grid alpha 0.35, dpi 300.

Three figures, because 36 curves on one axes is unreadable:

  1. cgyro/cgyro_by_separator.png      one panel per SEPARATOR (UniBin / Uniform /
     Deviation / Rho). Inside a panel only Budget and Select vary, so the panel
     answers "given this separator, does the selector matter?"
  2. cgyro/cgyro_separator_compare.png the Separate axis in isolation: Budget and
     Select pinned to Uniform/Random, so the ONLY thing differing between the four
     curves is how candidates are partitioned.
  3. cgyro/cgyro_selector_compare.png  the SELECT axis in isolation, the same idea
     one stage along: Separate and Budget pinned to UniBin/Uniform, so the only
     difference between the six curves is which scorer picks the batch. UniBin
     means no partitioning at all, so this is the cleanest look the study has at
     "does the acquisition scorer itself do anything?" -- and all six Select
     values exist at UniBin-Uniform, so the axis is complete with nothing dropped.
  4. cgyro/cgyro_selector_compare_all.png  figure 3 repeated once per SEPARATOR
     (Budget still pinned to Uniform), so the question becomes "is the Select
     ordering a property of the scorer, or of running it on an unpartitioned
     pool?". The five common Select values (Random/P-Flip/Importance/LCMD/EIG)
     exist under all four separators, so every panel is complete; DIRECT exists
     only at UniBin and appears in that panel alone. Colour is keyed to the
     SELECT value, not to rank, so a curve keeps its colour across panels --
     that is the whole point, and it is why this figure does not sort by final
     loss the way the other three do.

The paired-delta view of these same arms already exists for every strategy in
cgyro/cgyro_bootstrap_delta.png -- it is deliberately not duplicated here.

LABELS ARE COPIED VERBATIM FROM ACQ_GLOSSARY.md -- the Separate/Budget/Select
triple is NOT derivable from the config key (`res_rank_ow` is Deviation-Ranked-
Importance while `res_rank_ran` is Deviation-EIG-Random). PROJECT_KNOWLEDGE.md 15.7.
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bootstrap_data.json")
FIXED = os.path.join(HERE, "bootstrap_fixed_results.json")
OUTDIR = os.path.join(HERE, "cgyro")
BASELINE = "random"

# (Separate, Budget, Select) -- verbatim from ACQ_GLOSSARY.md
SPEC = {
    "random":          ("UniBin",    "Uniform", "Random"),
    "eig":             ("UniBin",    "Uniform", "EIG"),
    "glo_uni_ow":      ("UniBin",    "Uniform", "Importance"),
    "glo_uni_pflip":   ("UniBin",    "Uniform", "P-Flip"),
    "glo_uni_lcmd":    ("UniBin",    "Uniform", "LCMD"),
    "strat_uni_ran":   ("Uniform",   "Uniform", "Random"),
    "strat_uni_pflip": ("Uniform",   "Uniform", "P-Flip"),
    "strat_uni_ow":    ("Uniform",   "Uniform", "Importance"),
    "strat_uni_lcmd":  ("Uniform",   "Uniform", "LCMD"),
    "strat_rank_ran":  ("Uniform",   "EIG",     "Random"),
    "strat_rank_ow":   ("Uniform",   "Ranked",  "Importance"),
    "strat_rank_lcmd": ("Uniform",   "Ranked",  "LCMD"),
    "res_uni_ran":     ("Deviation", "Uniform", "Random"),
    "res_uni_pflip":   ("Deviation", "Uniform", "P-Flip"),
    "res_uni_ow":      ("Deviation", "Uniform", "Importance"),
    "res_uni_eig":     ("Deviation", "Uniform", "EIG"),
    "res_uni_lcmd":    ("Deviation", "Uniform", "LCMD"),
    "res_rank_ran":    ("Deviation", "EIG",     "Random"),
    "res_rank_pflip":  ("Deviation", "EIG",     "P-Flip"),
    "res_rank_ow":     ("Deviation", "Ranked",  "Importance"),
    "res_rank_lcmd":   ("Deviation", "Ranked",  "LCMD"),
    "rho_uni_ran":     ("Rho",       "Uniform", "Random"),
    "rho_uni_pflip":   ("Rho",       "Uniform", "P-Flip"),
    "rho_uni_ow":      ("Rho",       "Uniform", "Importance"),
    "rho_uni_eig":     ("Rho",       "Uniform", "EIG"),
    "rho_uni_lcmd":    ("Rho",       "Uniform", "LCMD"),
    "rho_rank_ran":    ("Rho",       "EIG",     "Random"),
    "rho_rank_ow":     ("Rho",       "Ranked",  "Importance"),
    # wave 5 mop-up + wave 6 -- completes all 36
    "rho_rank_lcmd":   ("Rho",       "EIG",     "LCMD"),
    "res_rank_eig":    ("Deviation", "EIG",     "EIG"),
    "direct":          ("UniBin",    "Uniform", "DIRECT"),
    "rho_rank_pflip":  ("Rho",       "EIG",     "P-Flip"),
    "strat_eig_pflip": ("Uniform",   "EIG",     "P-Flip"),
    "strat_uni_eig":   ("Uniform",   "Uniform", "EIG"),
    "eig_stratified":  ("Uniform",   "Ranked",  "EIG"),
    "rho_rank_eig":    ("Rho",       "EIG",     "EIG"),
}
LABEL = {k: "-".join(v) for k, v in SPEC.items()}
SEP_ORDER = ["UniBin", "Uniform", "Deviation", "Rho"]
# Random first: it is the baseline every delta in the study is measured against.
SEL_ORDER = ["Random", "P-Flip", "Importance", "LCMD", "EIG", "DIRECT"]


def load():
    d = json.load(open(DATA))
    usable = d["complete_case_splits"]
    cell = {(r["split_id"], r["acq"]): r for r in d["runs"]}
    acqs = [a for a in d["expect_acqs"] if a in SPEC]
    return usable, cell, acqs


def offline_ref(splits):
    # key is `full_test_loss` (test/run_bootstrap_fixed.py). Phase 1 logs it as a
    # single batch_size=1e6 eval while BAL averages per-batch means, so this is a
    # reference line, not a number to quote against BAL to 3 decimals.
    if not os.path.exists(FIXED):
        return None
    rows = [r for r in json.load(open(FIXED)) if r["split_id"] in set(splits)]
    vals = [r["full_test_loss"] for r in rows if "full_test_loss" in r]
    return float(np.mean(vals)) if vals else None


def curves(usable, cell, acqs):
    """Aggregate on ITERATION INDEX -- the only coordinate aligned across splits,
    since the initial train size varies with dedup -- and plot at the median n.
    Same convention as test/plot_bootstrap.py so the figures are comparable."""
    iters = sorted({p["iteration"] for s in usable for a in acqs
                    for p in cell[(s, a)]["points"]})
    out = {}
    for a in acqs:
        xs, mu = [], []
        for it in iters:
            vals = [p["loss"] for s in usable
                    for p in cell[(s, a)]["points"] if p["iteration"] == it]
            ns = [p["n"] for s in usable
                  for p in cell[(s, a)]["points"] if p["iteration"] == it]
            if len(vals) < 2:
                continue
            xs.append(np.median(ns))
            mu.append(float(np.mean(vals)))
        out[a] = (np.array(xs), np.array(mu))
    return out


def house(ax, xlabel=None, ylabel="Test Loss"):
    ax.set_xlabel("Total Cumulative Samples Acquired" if xlabel is None else xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.35)


def fig_by_separator(cur, ref):
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True, sharey=True)
    for ax, sep in zip(axes.ravel(), SEP_ORDER):
        keys = sorted([a for a in cur if SPEC[a][0] == sep],
                      key=lambda a: cur[a][1][-1])
        for i, a in enumerate(keys):
            x, y = cur[a]
            _, bud, sel = SPEC[a]
            ax.plot(x, y, label=f"{bud}-{sel}  ({y[-1]:.4f})", color=cmap(i % 20),
                    marker="o", markersize=3, linewidth=1.8, alpha=0.9)
        if ref is not None:
            ax.axhline(y=ref, color="k", ls=":", alpha=0.7,
                       label=f"offline baseline ({ref:.4f})")
        ax.set_title(f"{sep}  ({len(keys)} strategies)")
        # sharex hides the top row's tick labels, so only the bottom row gets an
        # x-label -- an axis title with no numbers under it reads as a bug
        house(ax, xlabel="" if ax in set(axes[0]) else None)
        ax.legend(fontsize=7.5, ncol=1, loc="upper right")
    fig.suptitle(
        f"CGYRO BAL Bootstrap by SEPARATOR: {len(cur)} acquisition functions, "
        "10 paired splits, 30 rounds x 50 new samples/round\n"
        "each panel fixes the Separate stage; legend = test loss at the final round",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, "cgyro_by_separator.png")
    fig.savefig(p, dpi=300)
    print("saved", p)


def fig_separator_compare(cur, ref):
    """Separate axis in isolation: Budget=Uniform, Select=Random for all four."""
    arms = sorted([a for a in cur if SPEC[a][1] == "Uniform" and SPEC[a][2] == "Random"],
                  key=lambda a: cur[a][1][-1])
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(14, 8.5))
    for i, a in enumerate(arms):
        x, y = cur[a]
        plt.plot(x, y, label=f"{SPEC[a][0]}  —  {LABEL[a]}  ({y[-1]:.4f})",
                 color=cmap(i % 20), marker="o", markersize=3, linewidth=1.8,
                 alpha=0.9)
    if ref is not None:
        plt.axhline(y=ref, color="k", ls=":", alpha=0.7,
                    label=f"offline baseline ({ref:.4f})")
    plt.xlabel("Total Cumulative Samples Acquired")
    plt.ylabel("Test Loss")
    plt.title("CGYRO BAL Bootstrap: the SEPARATOR axis in isolation\n"
              "Budget=Uniform and Select=Random for all four, so the only difference "
              "is how candidates are partitioned (legend = test loss at the final round)")
    plt.grid(True, which="both", ls="-", alpha=0.35)
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    p = os.path.join(OUTDIR, "cgyro_separator_compare.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print("saved", p)

    base = cur[BASELINE][1][-1]
    print("\nSeparator axis, final round (Budget=Uniform, Select=Random):")
    for a in sorted(arms, key=lambda a: SEP_ORDER.index(SPEC[a][0])):
        y = cur[a][1]
        print(f"  {SPEC[a][0]:<10} {LABEL[a]:<26} final {y[-1]:.4f}   "
              f"vs UniBin {y[-1] - base:+.4f}")


def fig_selector_compare(cur, ref):
    """Select axis in isolation: Separate=UniBin, Budget=Uniform for all six.

    UniBin is *no* partitioning and Uniform is *no* budget re-weighting, so the
    only thing differing between these curves is which scorer chooses the batch.
    All six Select values are present at UniBin-Uniform, so unlike the Separate
    axis this one is complete -- nothing is dropped for want of an arm.
    """
    arms = sorted([a for a in cur if SPEC[a][0] == "UniBin" and SPEC[a][1] == "Uniform"],
                  key=lambda a: cur[a][1][-1])
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(14, 8.5))
    for i, a in enumerate(arms):
        x, y = cur[a]
        plt.plot(x, y, label=f"{SPEC[a][2]}  —  {LABEL[a]}  ({y[-1]:.4f})",
                 color=cmap(i % 20), marker="o", markersize=3, linewidth=1.8,
                 alpha=0.9)
    if ref is not None:
        plt.axhline(y=ref, color="k", ls=":", alpha=0.7,
                    label=f"offline baseline ({ref:.4f})")
    house(plt.gca())
    plt.title("CGYRO BAL Bootstrap: the SELECT axis in isolation\n"
              "Separate=UniBin (no partitioning) and Budget=Uniform for all six, so "
              "the only difference is which scorer picks the batch\n"
              "(legend = test loss at the final round)")
    plt.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    p = os.path.join(OUTDIR, "cgyro_selector_compare.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print("saved", p)

    base = cur[BASELINE][1][-1]
    print("\nSelect axis, final round (Separate=UniBin, Budget=Uniform):")
    for a in sorted(arms, key=lambda a: SEL_ORDER.index(SPEC[a][2])):
        y = cur[a][1]
        print(f"  {SPEC[a][2]:<12} {LABEL[a]:<26} final {y[-1]:.4f}   "
              f"vs Random {y[-1] - base:+.4f}")



# --- figure 4 helpers -------------------------------------------------------
# Colour is pinned to the SELECT value rather than cycled by rank, so the same
# scorer is the same colour in all four panels. tab20 indices are hand-picked to
# stay in the house palette while keeping Random (the baseline) visually distinct.
SEL_COLOR = {
    "Random":     "tab20:0",
    "P-Flip":     "tab20:2",
    "Importance": "tab20:4",
    "LCMD":       "tab20:6",
    "EIG":        "tab20:8",
    "DIRECT":     "tab20:14",
}


def _sel_color(sel):
    return plt.get_cmap("tab20")(int(SEL_COLOR[sel].split(":")[1]))


def _aualc(points):
    """Normalized area under the AL curve over cumulative n -- identical to
    test/plot_bootstrap.py:aualc, the study's primary endpoint."""
    n = np.asarray([p["n"] for p in points], dtype=float)
    y = np.asarray([p["loss"] for p in points], dtype=float)
    o = np.argsort(n)
    n, y = n[o], y[o]
    if n[-1] == n[0]:
        return float(np.mean(y))
    return float(np.trapezoid(y, n) / (n[-1] - n[0]))


def paired_vs(cell, usable, a, base):
    """Paired per-split deltas of `a` against `base`. Returns
    (mean AUALC delta, AUALC wins, mean final delta, final wins, S).

    Paired at the SPLIT level: the split-to-split sd of a single arm is ~0.32
    (15.15) against a whole-study effect span of ~0.12, so the unpaired
    difference of two means here is meaningless. Never average the arms first.
    """
    da, df = [], []
    for s in usable:
        pa, pb = cell[(s, a)]["points"], cell[(s, base)]["points"]
        da.append(_aualc(pa) - _aualc(pb))
        fa = max(pa, key=lambda p: p["iteration"])["loss"]
        fb = max(pb, key=lambda p: p["iteration"])["loss"]
        df.append(fa - fb)
    da, df = np.array(da), np.array(df)
    return (float(da.mean()), int((da < 0).sum()),
            float(df.mean()), int((df < 0).sum()), len(da))


def fig_selector_compare_all(cur, ref, cell, usable):
    """Figure 3, once per separator: Budget=Uniform, Separate fixed per panel.

    Each panel's baseline is its OWN Random arm (UniBin->`random`,
    Uniform->`strat_uni_ran`, Deviation->`res_uni_ran`, Rho->`rho_uni_ran`), not
    the global `random`, because a delta against the global baseline would mix
    the Select effect with the Separate effect and neither could be read off.
    """
    panels = []
    for sep in SEP_ORDER:
        arms = [a for a in cur if SPEC[a][0] == sep and SPEC[a][1] == "Uniform"]
        arms.sort(key=lambda a: SEL_ORDER.index(SPEC[a][2]))
        base = next((a for a in arms if SPEC[a][2] == "Random"), None)
        panels.append((sep, arms, base))

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True, sharey=True)
    for ax, (sep, arms, base) in zip(axes.ravel(), panels):
        for a in arms:
            x, y = cur[a]
            sel = SPEC[a][2]
            # Random is the reference in every panel -- make it read as one.
            is_base = a == base
            ax.plot(x, y, label=f"{sel}  ({y[-1]:.4f})", color=_sel_color(sel),
                    marker="o", markersize=3,
                    linewidth=2.6 if is_base else 1.8,
                    linestyle="-" if is_base else "-",
                    zorder=3 if is_base else 2, alpha=0.95 if is_base else 0.9)
        if ref is not None:
            ax.axhline(y=ref, color="k", ls=":", alpha=0.7,
                       label=f"offline baseline ({ref:.4f})")
        ax.set_title(f"Separate = {sep}   ({len(arms)} selectors)")
        house(ax, xlabel="" if ax in set(axes[0]) else None)
        ax.legend(fontsize=8.5, loc="upper right", title="Select")
    fig.suptitle(
        "CGYRO BAL Bootstrap: the SELECT axis in isolation, under EVERY separator"
        "      —      Budget = Uniform\n"
        "10 paired splits, 30 rounds x 50 new samples/round; legend = test loss at the final round",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTDIR, "cgyro_selector_compare_all.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("saved", p)

    # Paired table, each panel against its own Random arm.
    print("\nSelect axis under each separator (Budget=Uniform), paired vs that "
          "panel's own Random arm:")
    print(f"  {'Separate':<10} {'Select':<12} {'final':>8} {'dFinal':>9} {'win':>6} "
          f"{'AUALC':>8} {'dAUALC':>9} {'win':>6}")
    order = {}
    for sep, arms, base in panels:
        if base is None:
            continue
        for a in arms:
            y = cur[a][1]
            au = float(np.mean([_aualc(cell[(s, a)]["points"]) for s in usable]))
            if a == base:
                print(f"  {sep:<10} {SPEC[a][2]:<12} {y[-1]:>8.4f} {'baseline':>9} "
                      f"{'-':>6} {au:>8.4f} {'baseline':>9} {'-':>6}")
                continue
            dA, wA, dF, wF, S = paired_vs(cell, usable, a, base)
            print(f"  {sep:<10} {SPEC[a][2]:<12} {y[-1]:>8.4f} {dF:>+9.4f} "
                  f"{wF:>3}/{S:<2} {au:>8.4f} {dA:>+9.4f} {wA:>3}/{S:<2}")
            order.setdefault(SPEC[a][2], []).append((sep, dA, dF))
        print()

    # Does a scorer's sign hold across separators? That, not any single number,
    # is what the four panels are for.
    print("Consistency of each scorer across the four separators "
          "(sign of the paired delta vs the panel's own Random):")
    for sel in SEL_ORDER:
        rows = order.get(sel, [])
        if len(rows) < 2:
            continue
        nA = sum(1 for _, dA, _ in rows if dA < 0)
        nF = sum(1 for _, _, dF in rows if dF < 0)
        print(f"  {sel:<12} better on AUALC in {nA}/{len(rows)} separators, "
              f"on final round in {nF}/{len(rows)}   "
              f"(mean dAUALC {np.mean([d for _, d, _ in rows]):+.4f}, "
              f"mean dFinal {np.mean([d for _, _, d in rows]):+.4f})")



def main():
    os.makedirs(OUTDIR, exist_ok=True)
    usable, cell, acqs = load()
    print(f"{len(acqs)} strategies x {len(usable)} complete-case splits")
    ref = offline_ref(usable)
    cur = curves(usable, cell, acqs)
    fig_by_separator(cur, ref)
    fig_separator_compare(cur, ref)
    fig_selector_compare(cur, ref)
    fig_selector_compare_all(cur, ref, cell, usable)


if __name__ == "__main__":
    main()
