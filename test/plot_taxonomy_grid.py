"""The acquisition design space as a grid: Separate x Budget x Select.

Every BAL strategy in this project is a three-stage pipeline (ACQ_GLOSSARY.md):

    candidates -> 1. SEPARATE -> 2. BUDGET -> 3. SELECT -> acquired points

so the design space is 4 Separate x 3 Budget x 6 Select = **72 cells, of which 36
are registered** in `STRATEGY_HANDLER` (`src/bal/DerivedAcq.py`). This figure is
the map of which 36, and it is the paper's answer to the prior work's single
EIG-vs-random ablation: that comparison moves along ONE axis of three and cannot
see, for instance, that the Separate stage dominates the Select stage.

Cells are shaded by the paired AUALC delta against `random` on CGYRO (negative =
better than random), so the same figure carries the structure and the result.
Unregistered cells are hatched, not blank, so "we did not run it" never reads as
"it did nothing".

**Two labelling traps, both load-bearing:**

* The Separate-Budget-Select triple is **NOT derivable from the config key** --
  `res_rank_ran` is Deviation-**EIG**-Random while `res_rank_ow` is
  Deviation-**Ranked**-Importance, because `_budget_ranked_weights` ranks strata
  by whichever score the selector uses. The SPEC below is copied verbatim from
  the ACQ_GLOSSARY.md master table. Do not regenerate it from the key.
* **"Uniform" means two unrelated things.** As a SEPARATE word it is
  `_separate_stratified_residual`: sort candidates by residual, cut into
  equal-size bins -- the *bins* are uniform, the ordering is the residual, and
  there is no EIG anywhere in it. As a BUDGET word it is `_budget_uniform`, an
  even split across whatever bins the separator made. The axis labels below spell
  this out rather than relying on the reader to disambiguate.

Reads `test/bootstrap_data.json`; writes `cgyro/cgyro_taxonomy_grid.png`.
"""
import os
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(HERE, "bootstrap_data.json")
OUTDIR = os.path.join(HERE, "cgyro")
OUT    = os.path.join(OUTDIR, "cgyro_taxonomy_grid.png")
BASELINE = "random"

# (Separate, Budget, Select) -- verbatim from ACQ_GLOSSARY.md.
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
SEP_ORDER = ["UniBin", "Uniform", "Deviation", "Rho"]
BUD_ORDER = ["Uniform", "Ranked", "EIG"]
SEL_ORDER = ["Random", "P-Flip", "Importance", "LCMD", "EIG", "DIRECT"]
# Spelled out on the axes because the one-word names collide across stages.
SEP_GLOSS = {"UniBin": "UniBin\n(no partition)",
             "Uniform": "Uniform\n(residual, equal-size bins)",
             "Deviation": "Deviation\n(residual Z-score class)",
             "Rho": "Rho\n(radial band)"}
BUD_GLOSS = {"Uniform": "BUDGET = Uniform\n(even split)",
             "Ranked": "BUDGET = Ranked\n(by selector score)",
             "EIG": "BUDGET = EIG\n(ranked by EIG)"}


def aualc(points):
    n = np.array([p["n"] for p in points], float)
    y = np.array([p["loss"] for p in points], float)
    o = np.argsort(n)
    n, y = n[o], y[o]
    return float(np.mean(y)) if n[-1] == n[0] else float(np.trapezoid(y, n) / (n[-1] - n[0]))


def deltas():
    """Mean paired AUALC delta vs `random`, per strategy."""
    d = json.load(open(DATA))
    usable = d["complete_case_splits"]
    cell = {(r["split_id"], r["acq"]): r for r in d["runs"]}
    out = {}
    for a in d["expect_acqs"]:
        vals = [aualc(cell[(s, a)]["points"]) - aualc(cell[(s, BASELINE)]["points"])
                for s in usable if (s, a) in cell]
        if vals:
            out[a] = float(np.mean(vals))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    D = deltas()
    by_cell = {v: k for k, v in SPEC.items()}
    lim = max(abs(v) for v in D.values())
    cmap = plt.get_cmap("RdBu")

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.8), sharey=True)
    # Headroom for a 3-line suptitle above 2-line panel titles; without it the
    # two overlap and bbox_inches="tight" will not separate them.
    fig.subplots_adjust(top=0.76)
    for ax, bud in zip(axes, BUD_ORDER):
        for r, sep in enumerate(SEP_ORDER):
            for c, sel in enumerate(SEL_ORDER):
                key = by_cell.get((sep, bud, sel))
                if key is None:
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#f2f1ef",
                                           edgecolor="white", lw=2, hatch="///"))
                    continue
                v = D.get(key)
                # Diverging map centred on 0 = random. Blue better, red worse.
                face = cmap(0.5 - 0.5 * v / lim) if v is not None else "#d8d6d2"
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face,
                                       edgecolor="white", lw=2))
                txt = f"{key}\n{v:+.3f}" if v is not None else key
                ax.text(c + .5, r + .5, txt, ha="center", va="center", fontsize=6.6,
                        color="#0b0b0b" if abs(v or 0) < 0.6 * lim else "white")
        ax.set_xlim(0, len(SEL_ORDER)); ax.set_ylim(0, len(SEP_ORDER))
        ax.set_xticks(np.arange(len(SEL_ORDER)) + .5)
        ax.set_xticklabels(SEL_ORDER, fontsize=9)
        ax.set_yticks(np.arange(len(SEP_ORDER)) + .5)
        ax.set_yticklabels([SEP_GLOSS[s] for s in SEP_ORDER], fontsize=8.5)
        ax.set_title(BUD_GLOSS[bud], fontsize=11)
        ax.set_xlabel("SELECT", fontsize=9.5)
        ax.invert_yaxis()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("SEPARATE", fontsize=9.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("paired AUALC delta vs `random`  (negative = better)", fontsize=9)
    fig.suptitle(
        f"The acquisition design space: Separate x Budget x Select\n"
        f"{len(SPEC)} of {len(SEP_ORDER) * len(BUD_ORDER) * len(SEL_ORDER)} cells are "
        "registered strategies (hatched = not registered); shading is the CGYRO paired "
        "AUALC delta over 10 leak-free splits\n"
        "'Uniform' is two different stages: as SEPARATE it means residual-sorted "
        "equal-size bins, as BUDGET an even split", fontsize=12, y=0.99)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)
    print(f"{len(SPEC)} registered of 72; AUALC delta range "
          f"{min(D.values()):+.4f} .. {max(D.values()):+.4f}")


if __name__ == "__main__":
    main()
