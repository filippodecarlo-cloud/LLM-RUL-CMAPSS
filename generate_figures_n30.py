"""
Generate honesty figures for the paper:
  - Figure 3b: scatter plots for n=30 configurations (FD001)
  - Figure 3c: histogram of distinct predicted values per configuration

Run from Anaconda Prompt:
    cd "K:\\Il mio Drive\\02 UNI\\Articoli\\2026 AI SI\\experiment"
    python generate_figures_n30.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

RESULTS = Path("results")
FIGS    = Path("figures")
FIGS.mkdir(exist_ok=True)

C_LLM = "#4CAF50"


def load_pairs(filename):
    """Return arrays of (true_rul, pred_rul) from a traces JSON file."""
    with open(RESULTS / filename, encoding="utf-8") as f:
        data = json.load(f)
    t = np.array([d["true_rul"] for d in data])
    p = np.array([d["pred_rul"] for d in data])
    return t, p


def rmse(t, p):
    return float(np.sqrt(np.mean((t - p) ** 2)))


def save_fig(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", format="pdf")
    fig.savefig(FIGS / f"{name}.png", format="png")


# ════════════════════════════════════════════════════════════
# FIGURE 3b — Scatter plots for n=30 configurations (FD001)
# ════════════════════════════════════════════════════════════
def fig_scatter_n30():
    configs = [
        ("Llama Zero-Shot\nn=30",      "traces_FD001_zero_shot_k0_n30.json"),
        ("Llama Few-Shot k=3\nn=30",   "traces_FD001_few_shot_k3_n30.json"),
        ("Llama Few-Shot k=5\nn=30",   "traces_FD001_few_shot_k5_n30.json"),
        ("Llama Few-Shot k=10\nn=30",  "traces_FD001_few_shot_k10_n30.json"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True, sharex=True)
    lim = [0, 130]

    for ax, (label, fname) in zip(axes, configs):
        t, p = load_pairs(fname)
        r = rmse(t, p)
        ax.scatter(t, p, alpha=0.65, s=45, color=C_LLM,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction", zorder=2)
        ax.fill_between(lim, [v - 20 for v in lim], [v + 20 for v in lim],
                        alpha=0.08, color="grey", label="±20 cycles band")
        ax.set_xlabel("True RUL (cycles)")
        label_clean = label.replace("\n", " ")
        ax.set_title(f"{label_clean}\nRMSE = {r:.1f}", fontweight="bold")
        ax.set_xlim(0, 130); ax.set_ylim(0, 130)
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Predicted RUL (cycles)")
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Predicted vs. True RUL — Llama 3.1 8B, NASA CMAPSS FD001 (100 engines, n=30)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3b_scatter_n30")
    print("Figure 3b (scatter n=30) saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 3c — Histogram of distinct predicted values per configuration
# ════════════════════════════════════════════════════════════
def fig_distinct_values():
    configs_fd001 = [
        ("ZS n=5",   "traces_FD001_zero_shot_k0.json"),
        ("ZS n=15",  "traces_FD001_zero_shot_k0_n15.json"),
        ("ZS n=30",  "traces_FD001_zero_shot_k0_n30.json"),
        ("k=3 n=5",  "traces_FD001_few_shot_k3.json"),
        ("k=3 n=15", "traces_FD001_few_shot_k3_n15.json"),
        ("k=3 n=30", "traces_FD001_few_shot_k3_n30.json"),
        ("k=5 n=5",  "traces_FD001_few_shot_k5.json"),
        ("k=5 n=15", "traces_FD001_few_shot_k5_n15.json"),
        ("k=5 n=30", "traces_FD001_few_shot_k5_n30.json"),
        ("k=10 n=5", "traces_FD001_few_shot_k10.json"),
        ("k=10 n=15","traces_FD001_few_shot_k10_n15.json"),
        ("k=10 n=30","traces_FD001_few_shot_k10_n30.json"),
    ]

    labels = []
    n_distinct = []
    top_share = []  # share of predictions in the top-1 most frequent value

    for label, fname in configs_fd001:
        _, p = load_pairs(fname)
        c = Counter(p.tolist())
        labels.append(label)
        n_distinct.append(len(c))
        top_share.append(100 * c.most_common(1)[0][1] / len(p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    # Left: number of distinct values
    bars1 = ax1.bar(labels, n_distinct, color=C_LLM, edgecolor="white", zorder=3)
    ax1.set_ylabel("Number of distinct predicted values\n(out of 100 engines)")
    ax1.set_title("Categorical collapse: how many values does the model use?",
                  fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="x", rotation=45)
    for b, v in zip(bars1, n_distinct):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v),
                 ha="center", fontsize=9)

    # Right: share of predictions in the top-1 value
    bars2 = ax2.bar(labels, top_share, color="#FF7043", edgecolor="white", zorder=3)
    ax2.set_ylabel("Share of predictions in single most common value (%)")
    ax2.set_title("Anchor strength: how concentrated is the model on one value?",
                  fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.axhline(50, color="grey", lw=0.8, ls=":", zorder=1)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", rotation=45)
    for b, v in zip(bars2, top_share):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                 ha="center", fontsize=9)

    fig.suptitle("Llama 3.1 8B as implicit classifier — FD001",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3c_distinct_values")
    print("Figure 3c (distinct values) saved.")

    # Print also a small text report
    print("\n--- Summary table (FD001) ---")
    print(f"{'Config':<12}{'Distinct':>10}{'Top-1 share':>15}")
    for l, n, s in zip(labels, n_distinct, top_share):
        print(f"{l:<12}{n:>10}{s:>14.1f}%")


if __name__ == "__main__":
    fig_scatter_n30()
    fig_distinct_values()
    print("\nDone. Output PNG/PDF in ./figures/")
