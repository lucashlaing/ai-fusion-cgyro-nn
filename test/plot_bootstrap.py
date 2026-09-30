"""Phase-2 aggregation: paired-delta plots + the ranking table.

Reads ``test/bootstrap_data.json`` (written by ``test/pull_bootstrap.py``).

The whole point is PAIRING. With sigma_split ~ 0.15 and sigma_eps ~ 0.025, an
unpaired contrast has SE 0.215/sqrt(S) and a paired one 0.0354/sqrt(S) -- 6x
smaller. Reaching SE 0.01 unpaired would need ~460 splits; paired needs ~13.
Every estimator below therefore works on ``D(a,s) = M(a,s) - M(random,s)``, and
the bootstrap resamples the SPLIT INDEX ONCE per replicate and applies it to all
strategies jointly. Resampling per cell destroys the pairing and is the single easiest
way to produce a confidently wrong answer here.

Primary endpoint: **AUALC**, the normalized area under the active-learning curve,
``trapz(loss, n) / (n[-1] - n[0])``. It uses all 15 points instead of 1 (~sqrt(15)
less noise) and is what active learning actually claims. Secondary: final-iteration
``BAL/test_loss``. Never the loop's ``test_loss`` -- single batch.

Three things kept honest on purpose:

* Repeated K-fold test sets overlap across repeats, so naive s^2/S understates
  variance. Intervals are widened by the Nadeau-Bengio factor
  ``sqrt(1 + S * n_test/n_train)`` = ``sqrt(1 + S*0.25)`` at 80/20. That second
  term does not shrink with S, which is why 10 splits is the right call and 20
  buys almost nothing.
* Friedman is a global gate first (no multiplicity cost, and it turns a negative
  into one sentence); only then exact Wilcoxon + Holm across the 5 contrasts.
  TOST at delta=0.05 lets a null be stated as practical equivalence.
* The MDE is printed next to every null. Never claim zero; bound the effect.
  Winner's curse: E[min of 6 noisy means] is biased low by ~1.27*SE, so nothing
  is called "best" from an unadjusted point estimate -- Holm p and CIs are shown
  for ALL strategies.

Missing data is complete-case AT THE SPLIT LEVEL: if any strategy is missing or short
on split s, s is dropped from every contrast and reported. Never impute; never
truncate everyone to min(iters), because shortfall is systematic (is_pool_empty()
fires earlier for strategies that dedupe more) and truncation would compare
strategies at different budgets.
"""
import os
import json

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === USER INPUT ===
IN_JSON     = "./test/bootstrap_data.json"
# Phase-1 offline reference: finetuning.py on the FULL 80% train half of each
# split. Averaged over the SAME splits Phase 2 uses, so the line is paired to
# the curves rather than borrowed from another design. Do NOT reuse the old
# HLINE=1.512 from test/plot_acq_sweep.py -- that is the leaky per-row split.
FIXED_JSON  = "./test/bootstrap_fixed_results.json"
OUT_CURVES  = "./test/cgyro/cgyro_bootstrap_curves.png"
OUT_DELTA   = "./test/cgyro/cgyro_bootstrap_delta.png"
OUT_MD      = "./test/cgyro/cgyro_bootstrap.md"
# Canonical Separate-Budget-Select decomposition, verbatim from the
# ACQ_GLOSSARY.md table (lines 70-107) -- do NOT re-derive it from the config
# key, the mapping is not regular: `res_rank_ran` is Deviation-**EIG**-Random
# while `res_rank_ow` is Deviation-**Ranked**-Importance, because the ranking
# metric follows whichever score the selector uses.
# (separate, budget, select) per key.
SPEC = {
    "random":          ("UniBin",    "Uniform", "Random"),
    "rho_uni_ran":     ("Rho",       "Uniform", "Random"),
    "strat_uni_ran":   ("Uniform",   "Uniform", "Random"),
    "res_uni_ran":     ("Deviation", "Uniform", "Random"),
    "res_rank_ran":    ("Deviation", "EIG",     "Random"),
    "glo_uni_lcmd":    ("UniBin",    "Uniform", "LCMD"),
    "strat_uni_lcmd":  ("Uniform",   "Uniform", "LCMD"),
    "res_uni_lcmd":    ("Deviation", "Uniform", "LCMD"),
    "glo_uni_pflip":   ("UniBin",    "Uniform", "P-Flip"),
    "strat_uni_pflip": ("Uniform",   "Uniform", "P-Flip"),
    "res_uni_pflip":   ("Deviation", "Uniform", "P-Flip"),
    "res_rank_pflip":  ("Deviation", "EIG",     "P-Flip"),
    # batch 3 -- completes the Select axis
    "glo_uni_ow":      ("UniBin",    "Uniform", "Importance"),
    "strat_uni_ow":    ("Uniform",   "Uniform", "Importance"),
    "res_uni_ow":      ("Deviation", "Uniform", "Importance"),
    "res_rank_ow":     ("Deviation", "Ranked",  "Importance"),
    "eig":             ("UniBin",    "Uniform", "EIG"),
    "res_uni_eig":     ("Deviation", "Uniform", "EIG"),
    # wave 4 (2026-09-10) -- completes the Rho separator row
    "rho_uni_pflip":   ("Rho",       "Uniform", "P-Flip"),
    "rho_uni_ow":      ("Rho",       "Uniform", "Importance"),
    "rho_uni_lcmd":    ("Rho",       "Uniform", "LCMD"),
    "rho_uni_eig":     ("Rho",       "Uniform", "EIG"),
    "rho_rank_ran":    ("Rho",       "EIG",     "Random"),
    "rho_rank_ow":     ("Rho",       "Ranked",  "Importance"),
    # wave 5 (2026-09-12) -- the Ranked/EIG budget axis
    "strat_rank_ran":  ("Uniform",   "EIG",     "Random"),
    "strat_rank_ow":   ("Uniform",   "Ranked",  "Importance"),
    "strat_rank_lcmd": ("Uniform",   "Ranked",  "LCMD"),
    "res_rank_lcmd":   ("Deviation", "Ranked",  "LCMD"),
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
# House style is test/plot_pool_continue.py -> acq_comparison_pool_continue.png:
# 14x8.5 at dpi 300, tab20 cycled, 'o' markers ms=3, lw 1.8, legend inside upper
# right in two columns with each curve's final loss in parentheses, sorted by
# that value, dotted black offline baseline.
FIGSIZE   = (14, 8.5)
INK, INK_2 = "#0b0b0b", "#52514e"
LABELS = {k: f"{sep}-{bud}-{sel}" for k, (sep, bud, sel) in SPEC.items()}
B_BOOT      = 10000
ALPHA       = 0.05
TOST_DELTA  = 0.05          # practical-equivalence margin, loss units
NB_RATIO    = 0.25          # n_test/n_train at 80/20 -- the Nadeau-Bengio term
BOOT_SEED   = 20260831
# ==================


# ----------------------------------------------------------------------
def aualc(points):
    """Normalized area under the active-learning curve, over cumulative n."""
    n = np.asarray([p["n"] for p in points], dtype=float)
    y = np.asarray([p["loss"] for p in points], dtype=float)
    o = np.argsort(n)
    n, y = n[o], y[o]
    if n[-1] == n[0]:
        return float(np.mean(y))
    return float(np.trapezoid(y, n) / (n[-1] - n[0]))


def offline_baseline(path, splits):
    """Mean Phase-1 full-data test loss over exactly `splits`.

    finetuning.py logs FULL TEST LOSS as ONE batch (batch_size=1e6) while BAL
    logs BAL/test_loss over cfg.batch=256 batches and averages per-batch means,
    so the two estimators differ slightly on the runt batch. Fine as a reference
    line; do not quote the gap to 3 decimals.
    """
    if not os.path.exists(path):
        return None, None
    rows = [r for r in json.load(open(path)) if r["split_id"] in set(splits)]
    if not rows:
        return None, None
    v = np.array([r["full_test_loss"] for r in rows], dtype=float)
    return float(v.mean()), len(v)


def joint_bca(D, strategies, stat, B, alpha, seed):
    """BCa over splits, resampling the SPLIT INDEX ONCE per replicate.

    `D` is {strategy: array over splits}. All strategies see the same resampled index, so
    the pairing survives. Resampling each strategy independently would destroy it.
    """
    rng = np.random.default_rng(seed)
    S = len(next(iter(D.values())))
    idx_boots = [rng.integers(0, S, S) for _ in range(B)]
    out = {}
    for a in strategies:
        d = np.asarray(D[a], dtype=float)
        theta = stat(d)
        boots = np.array([stat(d[i]) for i in idx_boots])
        prop = min(max(np.mean(boots < theta), 1.0 / B), 1 - 1.0 / B)
        z0 = stats.norm.ppf(prop)
        jack = np.array([stat(np.delete(d, i)) for i in range(S)])
        jm = jack.mean()
        num = ((jm - jack) ** 3).sum()
        den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
        acc = num / den if den != 0 else 0.0
        lohi = []
        for q in (alpha / 2, 1 - alpha / 2):
            z = stats.norm.ppf(q)
            adj = z0 + (z0 + z) / max(1e-12, (1 - acc * (z0 + z)))
            lohi.append(float(np.percentile(boots, 100 * stats.norm.cdf(adj))))
        out[a] = (float(theta), lohi[0], lohi[1])
    return out


def nb_widen(center, lo, hi, S, ratio=NB_RATIO):
    """Nadeau-Bengio correction: Var = (1/S + n_test/n_train) * s^2.

    Relative to the naive 1/S, that is a factor sqrt(1 + S*ratio) on the width,
    and the ratio term does NOT shrink with S.
    """
    f = np.sqrt(1.0 + S * ratio)
    return center - (center - lo) * f, center + (hi - center) * f


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def mde(d, S, alpha=ALPHA, power=0.8, ratio=NB_RATIO, n_tests=1):
    """Minimum detectable effect for a paired test on S splits, NB-corrected."""
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    if not np.isfinite(sd):
        return np.nan
    se = sd * np.sqrt(1.0 / S + ratio)
    za = stats.norm.ppf(1 - alpha / (2 * max(1, n_tests)))
    zb = stats.norm.ppf(power)
    return (za + zb) * se


# ----------------------------------------------------------------------
def main():
    payload = json.load(open(IN_JSON))
    runs = payload["runs"]
    baseline = payload["baseline"]
    strategies = payload["expect_acqs"]
    expect_iters = payload["expect_iters"]

    cell = {(r["split_id"], r["acq"]): r for r in runs}
    splits = sorted({r["split_id"] for r in runs})

    # complete-case AT SPLIT LEVEL -- never impute, never truncate to min(iters)
    usable, dropped_splits = [], []
    for s in splits:
        if all((s, a) in cell and len(cell[(s, a)]["points"]) >= expect_iters
               for a in strategies):
            usable.append(s)
        else:
            miss = [a for a in strategies
                    if (s, a) not in cell or len(cell[(s, a)]["points"]) < expect_iters]
            dropped_splits.append((s, miss))
    S = len(usable)
    assert S >= 3, f"only {S} complete-case splits; need >=3. dropped={dropped_splits}"

    # ---- endpoints ---------------------------------------------------------
    A = {a: np.array([aualc(cell[(s, a)]["points"]) for s in usable]) for a in strategies}
    F = {a: np.array([cell[(s, a)]["points"][-1]["loss"] for s in usable]) for a in strategies}
    D_a = {a: A[a] - A[baseline] for a in strategies}
    D_f = {a: F[a] - F[baseline] for a in strategies}
    contrasts = [a for a in strategies if a != baseline]

    # ---- iteration-0 falsification gate ------------------------------------
    # Iteration 0 is BEFORE any acquisition: audit (a2) proves all strategies
    # hold the identical initial training set there, so D@iter0 must be 0. If it
    # is not, something that is NOT the acquisition function is moving the loss
    # (dropout mode per PROJECT_KNOWLEDGE 13.5, RNG consumption, eval-mode leak),
    # and AUALC -- which integrates that offset -- is measuring it. Correlating
    # D@iter0 against the AUALC delta says outright how much of the "result" is
    # the artefact.
    D_0 = {a: np.array([cell[(s_, a)]["points"][0]["loss"]
                        - cell[(s_, baseline)]["points"][0]["loss"]
                        for s_ in usable]) for a in strategies}
    d0_mu = np.array([D_0[a].mean() for a in contrasts])
    da_mu = np.array([D_a[a].mean() for a in contrasts])
    df_mu = np.array([D_f[a].mean() for a in contrasts])
    r_iter0_aualc = float(np.corrcoef(d0_mu, da_mu)[0, 1]) if len(contrasts) > 2 else np.nan
    r_iter0_final = float(np.corrcoef(d0_mu, df_mu)[0, 1]) if len(contrasts) > 2 else np.nan
    worst_d0 = float(np.abs(d0_mu).max())
    ITER0_TOL = 0.02
    iter0_broken = worst_d0 > ITER0_TOL

    # ---- global gate: Friedman across all strategies -----------------------------
    fr_stat, fr_p = stats.friedmanchisquare(*[A[a] for a in strategies])

    # ---- paired BCa (joint resample) + NB widening -------------------------
    med = lambda x: float(np.median(x))
    bca_med = joint_bca(D_a, contrasts, med, B_BOOT, ALPHA, BOOT_SEED)
    bca_mean = joint_bca(D_a, contrasts, lambda x: float(np.mean(x)),
                         B_BOOT, ALPHA, BOOT_SEED)

    # ---- exact Wilcoxon + Holm --------------------------------------------
    raw_p, tost_p = [], []
    for a in contrasts:
        d = D_a[a]
        if np.allclose(d, 0):
            raw_p.append(1.0)
        else:
            mode = "exact" if S <= 25 else "approx"
            raw_p.append(float(stats.wilcoxon(d, method=mode).pvalue))
        # TOST at +/- TOST_DELTA (paired t on each one-sided bound)
        t1 = stats.ttest_1samp(d + TOST_DELTA, 0, alternative="greater").pvalue
        t2 = stats.ttest_1samp(d - TOST_DELTA, 0, alternative="less").pvalue
        tost_p.append(float(max(t1, t2)))
    holm_p = holm(np.array(raw_p))

    # ---- table -------------------------------------------------------------
    lines = []
    lines.append("# CGYRO bootstrap BAL study (group K-fold, paired)\n")
    lines.append(f"- splits used (complete case): **{S}** -> {usable}")
    if dropped_splits:
        lines.append(f"- splits dropped: {dropped_splits} (complete-case at split "
                     f"level; nothing imputed, nothing truncated)")
    lines.append(f"- baseline strategy: `{baseline}`; primary endpoint **AUALC** "
                 f"(normalized area under the AL curve over cumulative n)")
    lines.append(f"- CIs are paired BCa (B={B_BOOT}, split index resampled ONCE per "
                 f"replicate across all strategies) then widened by the Nadeau-Bengio "
                 f"factor sqrt(1 + S*{NB_RATIO}) = {np.sqrt(1 + S * NB_RATIO):.2f}")
    lines.append(f"- **numbers here are on a different scale from every pre-existing "
                 f"CGYRO figure** (1.512 / 1.7548 / the 36-way sweep): those used the "
                 f"leaky per-row split. Never mix them in one table.\n")
    if iter0_broken:
        lines.append(
            f"> **STOP -- the primary endpoint is contaminated.** Iteration 0 is "
            f"before any acquisition and every strategy provably holds the same "
            f"initial training set there (audit a2), so the paired delta at "
            f"iteration 0 must be 0. It is not: the largest is "
            f"**{worst_d0:+.4f}**, and across the {len(contrasts)} contrasts the "
            f"iteration-0 offset correlates **r={r_iter0_aualc:+.3f}** with the "
            f"AUALC delta below. AUALC integrates that offset, so the ranking is "
            f"largely a ranking of a non-acquisition artefact. The final-iteration "
            f"endpoint is far less exposed (r={r_iter0_final:+.3f}) -- read that "
            f"one.\n>\n"
            f"> **A mechanism that is present in the code** (its share of the "
            f"observed scatter is NOT yet established -- see the note below). "
            f"`Spectra_Regularization_Trainer.get_test_loss` returns "
            f"`sum(losses)/len(losses)` -- an unweighted mean over BATCHES, not "
            f"rows -- and `bal_finetune.py:100-106` builds the test loader at "
            f"`batch_size=cfg.batch` (256). At ~551 test rows the batches are "
            f"256/256/39, so the 39-row runt carries **1/3 of the weight while "
            f"holding 7% of the rows**, and the datapipe reshuffles every pass, "
            f"so which rows land in it changes run to run. Measured spread of "
            f"that noise: sd ~0.22. `src/finetuning.py:163-174` already avoids "
            f"it with `batch_size=10**6` (one batch = the exact row mean), which "
            f"is why the offline reference line has no such scatter.\n>\n"
            f"> Caveat: the falsifiable prediction that follows -- splits with a "
            f"smaller runt batch should scatter more -- comes out r=+0.14, "
            f"p=0.70 over the 10 splits. Right sign, no power (each sd is "
            f"estimated from 12 values). So the mis-weighting is real "
            f"arithmetic, but whether it dominates the 0.22 is unproven; a "
            f"second candidate is RNG divergence changing the shuffle order "
            f"itself. Either way the same one-line change removes it. Fix "
            f"`bal_finetune.py` and re-run before quoting any row below.\n")
        lines.append("| strategy | mean D at iteration 0 (must be 0) |")
        lines.append("|---|---|")
        for a in contrasts:
            lines.append(f"| `{a}` | {D_0[a].mean():+.4f} |")
        lines.append("")
    lines.append(f"**Friedman global gate across {len(strategies)} strategies: "
                 f"chi2={fr_stat:.3f}, p={fr_p:.4f}** "
                 f"({'gate passed -- pairwise tests below are interpretable' if fr_p < ALPHA else 'gate NOT passed -- pairwise tests below are exploratory only'})\n")

    lines.append("| strategy | AUALC mean | median D | BCa CI (NB) | mean D | Wilcoxon p "
                 "| Holm p | TOST p (d=%.2f) | wins/S |" % TOST_DELTA)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| `{baseline}` (baseline) | {A[baseline].mean():.4f} | 0 | - | 0 "
                 f"| - | - | - | - |")
    order = sorted(range(len(contrasts)), key=lambda i: bca_med[contrasts[i]][0])
    for i in order:
        a = contrasts[i]
        m, lo, hi = bca_med[a]
        lo, hi = nb_widen(m, lo, hi, S)
        mu = bca_mean[a][0]
        wins = int((D_a[a] < 0).sum())
        lines.append(f"| `{a}` | {A[a].mean():.4f} | {m:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {mu:+.4f} | {raw_p[i]:.4f} | "
                     f"{holm_p[i]:.4f} | {tost_p[i]:.4f} | {wins}/{S} |")
    lines.append("\n(Negative D = better than random. Ordering is by point estimate "
                 "and is NOT a claim: E[min of %d noisy means] is biased low by "
                 "~1.27*SE, so no strategy is called best without a Holm-significant "
                 "interval.)\n" % len(strategies))

    # secondary endpoint
    lines.append("## Secondary endpoint: final-iteration BAL/test_loss\n")
    lines.append("| strategy | final mean | median D | wins/S |")
    lines.append("|---|---|---|---|")
    lines.append(f"| `{baseline}` | {F[baseline].mean():.4f} | 0 | - |")
    for a in contrasts:
        lines.append(f"| `{a}` | {F[a].mean():.4f} | {np.median(D_f[a]):+.4f} | "
                     f"{int((D_f[a] < 0).sum())}/{S} |")

    # MDE next to any null
    lines.append("\n## Minimum detectable effect (print this next to every null)\n")
    lines.append(f"| strategy | sd(D) | MDE (alpha={ALPHA}, power 0.8, Holm m={len(contrasts)}) |")
    lines.append("|---|---|---|")
    for a in contrasts:
        lines.append(f"| `{a}` | {np.std(D_a[a], ddof=1):.4f} | "
                     f"{mde(D_a[a], S, n_tests=len(contrasts)):.4f} |")
    lines.append("\nFor scale: the whole 36-way CGYRO spread was 0.092, best-vs-random "
                 "0.0505, and changing training length was worth 0.27. An effect below "
                 "the MDE is *bounded*, not zero -- say so.\n")
    n_cells = len(contrasts) * len(
        {p["iteration"] for s_ in usable for p in cell[(s_, baseline)]["points"]})
    lines.append(f"Anything not in the pre-declared primary endpoint (AUALC on the "
                 f"complete-case splits) is exploratory: {len(contrasts)} contrasts x "
                 f"{n_cells // max(1, len(contrasts))} iterations = {n_cells} cells "
                 f"yields ~{0.05 * n_cells:.0f} 'significant' results under the null "
                 f"at alpha=0.05 -- exactly the mechanism behind 23-of-36 apparent "
                 f"winners.\n")

    os.makedirs(os.path.dirname(os.path.abspath(OUT_MD)), exist_ok=True)
    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_MD}")

    # ---- plots -------------------------------------------------------------
    off, n_off = offline_baseline(FIXED_JSON, usable)
    if off is not None:
        print(f"\noffline reference (Phase 1, full 80% train, same {n_off} "
              f"splits): {off:.4f}")

    # Aggregate on ITERATION INDEX -- the only coordinate aligned across splits,
    # since the initial train size varies with dedup -- and plot at the median n.
    iters = sorted({p["iteration"] for s_ in usable for a in strategies
                    for p in cell[(s_, a)]["points"]})

    def curve(a):
        xs, mu = [], []
        for it in iters:
            vals = [p["loss"] for s_ in usable
                    for p in cell[(s_, a)]["points"] if p["iteration"] == it]
            ns = [p["n"] for s_ in usable
                  for p in cell[(s_, a)]["points"] if p["iteration"] == it]
            if len(vals) < 2:
                continue
            xs.append(np.median(ns))
            mu.append(float(np.mean(vals)))
        return xs, mu

    cmap = plt.get_cmap("tab20")

    def draw(ax, keys, series, ylabel):
        """One axes, reference house style, legend carrying each final value."""
        order = sorted(keys, key=lambda a: series[a][1][-1])
        for i, a in enumerate(order):
            x, y = series[a]
            ax.plot(x, y, color=cmap(i % 20), marker="o", markersize=3,
                    linewidth=1.8, alpha=0.9,
                    label=f"{LABELS[a]}  ({y[-1]:.4f})")
        ax.set_xlabel("Total Cumulative Samples Acquired")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", ls="-", alpha=0.35)

    # --- absolute curves ----------------------------------------------------
    series = {a: curve(a) for a in strategies}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw(ax, list(strategies), series, "Test Loss")
    if off is not None:
        ax.axhline(y=off, color="k", ls=":", alpha=0.7,
                   label=f"offline baseline ({off:.4f})")
    ax.set_title(f"CGYRO BAL Bootstrap: {len(strategies)} Acquisition Functions, "
                 f"30 rounds x 50 new samples/round")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_CURVES, dpi=300)
    plt.close(fig)
    for a in strategies:
        print(f"{LABELS[a]:32s} final {series[a][1][-1]:.4f} "
              f"at {series[a][0][-1]:.0f} samples")
    print(f"Saved plot to {OUT_CURVES}")

    # --- THE important one: paired delta vs baseline ------------------------
    # Split variance cancels here, so this is the only view where the strategies
    # are actually distinguishable -- and where the iteration-0 offset the gate
    # above flags shows up as a non-zero intercept.
    base_x = {it: x for it, x in zip(iters, curve(baseline)[0])}

    def delta(a):
        xs, mu = [], []
        for it in iters:
            d = []
            for s_ in usable:
                pa = {p["iteration"]: p["loss"] for p in cell[(s_, a)]["points"]}
                pb = {p["iteration"]: p["loss"] for p in cell[(s_, baseline)]["points"]}
                if it in pa and it in pb:
                    d.append(pa[it] - pb[it])
            if len(d) < 2 or it not in base_x:
                continue
            xs.append(base_x[it]); mu.append(float(np.mean(d)))
        return xs, mu

    dseries = {a: delta(a) for a in contrasts}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw(ax, contrasts, dseries,
         f"Test Loss minus {LABELS[baseline]}  (negative = better)")
    ax.axhline(0, color="k", lw=1.2, zorder=4)
    if iter0_broken:
        ax.annotate(f"at the left edge no acquisition has happened yet and every "
                    f"strategy holds the SAME initial set,\nso these gaps (up to "
                    f"{worst_d0:.2f}) are measurement error, not an acquisition effect",
                    xy=(0.015, 0.02), xycoords="axes fraction",
                    fontsize=8.5, color=INK_2, va="bottom")
    ax.set_title(f"CGYRO BAL Bootstrap: Paired Delta vs {LABELS[baseline]}, "
                 f"{S} splits\n"
                 f"legend = delta at the final round; split variance cancels")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DELTA, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {OUT_DELTA}")


if __name__ == "__main__":
    main()
