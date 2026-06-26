"""
Visualize the per-feature comparison in test/feature_stats_comparison.csv.

Writes two PNGs into test/plots/:
  1. feature_stats_per_feature.png  -- one tile per feature, showing JSON
     and pool side-by-side: full range as a thin line, mean +/- std as a
     thick bar, mean as a dot, min/max as ticks. Different x-axis per tile.
  2. feature_stats_summary.png      -- two ranked bar charts of
     mean_shift_in_json_stds and std_ratio_pool_over_json.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = os.path.join(REPO_ROOT, "test/feature_stats_comparison.csv")
PLOT_DIR = os.path.join(REPO_ROOT, "test/plots")
os.chdir(REPO_ROOT)
os.makedirs(PLOT_DIR, exist_ok=True)


def load_rows():
    with open(CSV_IN, "r") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def plot_per_feature(rows):
    n = len(rows)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.2 * nrows))
    for ax, r in zip(axes.flat, rows):
        j_mean = float(r["json_mean"]); j_std = float(r["json_std"])
        j_min = float(r["json_min"]);   j_max = float(r["json_max"])
        p_mean = float(r["pool_mean"]); p_std = float(r["pool_std"])
        p_min = float(r["pool_min"]);   p_max = float(r["pool_max"])

        # Pool at y=0, JSON at y=1
        ax.plot([p_min, p_max], [0, 0], color="tab:blue", alpha=0.35, lw=1.5)
        ax.plot([p_mean - p_std, p_mean + p_std], [0, 0],
                color="tab:blue", lw=8, alpha=0.65, solid_capstyle="butt")
        ax.plot(p_mean, 0, "o", color="navy", ms=6, zorder=5)
        ax.plot([p_min, p_min], [-0.1, 0.1], color="tab:blue", lw=1)
        ax.plot([p_max, p_max], [-0.1, 0.1], color="tab:blue", lw=1)

        ax.plot([j_min, j_max], [1, 1], color="tab:orange", alpha=0.35, lw=1.5)
        ax.plot([j_mean - j_std, j_mean + j_std], [1, 1],
                color="tab:orange", lw=8, alpha=0.65, solid_capstyle="butt")
        ax.plot(j_mean, 1, "o", color="darkorange", ms=6, zorder=5)
        ax.plot([j_min, j_min], [0.9, 1.1], color="tab:orange", lw=1)
        ax.plot([j_max, j_max], [0.9, 1.1], color="tab:orange", lw=1)

        # Sentinel marker: many JSON entries have max = 2.4472 (clip).
        if abs(j_max - 2.4472) < 1e-3:
            ax.annotate("clip", xy=(j_max, 1), xytext=(2, 6),
                        textcoords="offset points", fontsize=6,
                        color="firebrick")

        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["pool", "json"], fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        z = float(r["mean_shift_in_json_stds"])
        ratio = float(r["std_ratio_pool_over_json"])
        ax.set_title(f"{r['feature']}   z={z:.2f}  ratio={ratio:.2f}",
                     fontsize=9)
        ax.grid(axis="x", alpha=0.2)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Per-feature: JSON (aggregated) vs pool   "
                 "[thin line = range, thick bar = mean ± std, dot = mean]",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_stats_per_feature.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_summary(rows):
    feats = [r["feature"] for r in rows]
    z = [float(r["mean_shift_in_json_stds"]) for r in rows]
    ratio = [float(r["std_ratio_pool_over_json"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    order1 = sorted(range(len(feats)), key=lambda i: -z[i])
    ax1.barh([feats[i] for i in order1], [z[i] for i in order1],
             color="steelblue")
    ax1.invert_yaxis()
    ax1.set_xlabel("|pool_mean - json_mean| / json_std")
    ax1.set_title("Mean shift (smaller = closer)")
    ax1.axvline(0, color="k", lw=0.5)
    ax1.grid(axis="x", alpha=0.3)

    order2 = sorted(range(len(feats)), key=lambda i: -abs(ratio[i] - 1.0))
    ax2.barh([feats[i] for i in order2], [ratio[i] for i in order2],
             color="coral")
    ax2.invert_yaxis()
    ax2.set_xlabel("pool_std / json_std")
    ax2.set_title("Std ratio (1.0 = identical)")
    ax2.axvline(1, color="k", lw=0.5)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Per-feature summary: pool vs JSON-aggregated", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_stats_summary.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {CSV_IN}")
    plot_per_feature(rows)
    plot_summary(rows)


if __name__ == "__main__":
    main()
