import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import scienceplots

plt.style.use(['science','no-latex', 'high-vis'])


# ── helpers ──────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "inference_metric",
    "robustness_metric",
    "temporal_metric",
    "performance_metric",
]

METRIC_LABELS = {
    "performance_metric": "Performance",
    "inference_metric":   "Inference",
    "temporal_metric":    "Temporal",
    "robustness_metric":  "Robustness",
}

_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

CORR_STYLES = {
    "Pearson r":  {"color": _cycle[0]},
    "Spearman ρ": {"color": _cycle[1]},
    # "Kendall τ":  {"color": _cycle[2]}
}


# Hardcoded runtime configuration.
CSV_PATH = "ranking_mean.csv"
RANK_COL = "rank"
OUTPUT_PATH = None
PLOT_TITLE = "Metric Rank Correlation"


def compute_correlations(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Return a DataFrame (metrics × corr_types) of correlation values."""
    results = {}
    for metric in METRIC_COLS:
        x = df[metric].values
        y = df[rank_col].values
        results[metric] = {
            "Pearson r":  stats.pearsonr(x, y).statistic,
            "Spearman ρ": stats.spearmanr(x, y).statistic,
            # "Kendall τ":  stats.kendalltau(x, y).statistic,
        }
    return pd.DataFrame(results).T


def tornado_plot(
    corr_df: pd.DataFrame,
    title: str = "Metric–Rank Correlation",
    output_path: str | None = None,
) -> None:
    """Draw a grouped horizontal bar tornado plot from a correlation DataFrame."""

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    corr_types = corr_df.columns.tolist()   # e.g. ["Pearson r", "Spearman ρ"]
    metrics    = corr_df.index.tolist()

    n_metrics = len(metrics)
    n_types   = len(corr_types)
    bar_h     = 0.22
    group_gap = 0.08
    group_h   = n_types * bar_h + group_gap

    fig, ax = plt.subplots(figsize=(10, 4))

    y_positions = np.arange(n_metrics, dtype=float) * group_h
    offsets     = np.linspace(
        -(n_types - 1) / 2 * bar_h,
         (n_types - 1) / 2 * bar_h,
        n_types,
    )

    patches = []
    for j, ctype in enumerate(corr_types):
        style  = CORR_STYLES[ctype]
        values = corr_df[ctype].values
        ys     = y_positions + offsets[j]

        for i, (v, y) in enumerate(zip(values, ys)):
            color = style["color"]
            ax.barh(
                y, v,
                height=bar_h - 0.02,
                color=color + "cc",          # slight transparency
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            # value label
            xpad  = 0.015 * np.sign(v) if v != 0 else 0.015
            ha    = "left" if v >= 0 else "right"
            ax.text(
                v + xpad, y, f"{v:+.3f}",
                va="center", ha=ha, fontsize=10,
                color="#333333",
            )

        patches.append(
            mpatches.Patch(
                facecolor=style["color"] + "cc",
                edgecolor=style["color"],
                label=ctype,
            )
        )

    # ── axes decoration ───────────────────────────────────────────────────────
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [METRIC_LABELS.get(m, m) for m in metrics],
        fontsize=14,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Correlation With Final Rank", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)

    ax.legend(
        handles=patches,
        loc="best",
        fontsize=10,
        edgecolor="#cccccc",
        frameon=True,
    )

    plt.tight_layout()
    plt.grid()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # normalise column names
    df.columns = df.columns.str.strip()

    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in CSV: {missing}")

    if RANK_COL not in df.columns:
        print(
            f"Column '{RANK_COL}' not found — computing rank as mean of metric columns."
        )
        df[RANK_COL] = df[METRIC_COLS].mean(axis=1)

    corr_df = compute_correlations(df, RANK_COL)

    print("\nCorrelation table:")
    print(corr_df.round(4).to_string())
    print()

    tornado_plot(corr_df, title=PLOT_TITLE, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()