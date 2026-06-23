"""
Generate all paper figures from experiment results.
Run from Anaconda Prompt:
    pip install matplotlib seaborn
    python generate_figures.py

Outputs both PDF (print-quality) and PNG (preview) into ./figures/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

RESULTS = Path("results")
FIGS    = Path("figures")
FIGS.mkdir(exist_ok=True)

# ── Colour palette ───────────────────────────────────────────
C_RF   = "#2196F3"   # blue
C_LSTM = "#FF9800"   # orange
C_LLM  = "#4CAF50"   # green
C_GREY = "#9E9E9E"

# ── Baseline values (full 100-engine evaluation) ─────────────
RF_RMSE_FD001,   RF_MAE_FD001   = 17.96, 13.27
LSTM_RMSE_FD001, LSTM_MAE_FD001 = 43.70, 39.23


def save_fig(fig, name):
    """Save both PDF and PNG."""
    fig.savefig(FIGS / f"{name}.pdf", format="pdf")
    fig.savefig(FIGS / f"{name}.png", format="png")
    plt.close(fig)


# ── Load traces ──────────────────────────────────────────────
def load_traces(path):
    with open(path) as f:
        data = json.load(f)
    true = np.array([d["true_rul"] for d in data])
    pred = np.array([d["pred_rul"] for d in data], dtype=float)
    return true, pred

def metric_rmse(t, p): return float(np.sqrt(np.mean((p - t)**2)))
def metric_mae(t, p):  return float(np.mean(np.abs(p - t)))
def metric_nasa(t, p):
    d = p - t
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))


# ════════════════════════════════════════════════════════════
# FIGURE 1 — Conceptual prompting framework
# ════════════════════════════════════════════════════════════
def fig_framework():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")

    # Boxes: vertically narrower, leaving clear space above for arrow labels
    BOX_Y, BOX_H = 0.05, 0.60      # boxes occupy y=0.05..0.65
    ARROW_Y      = 0.35             # mid of box vertical range
    LABEL_Y      = 0.85             # well above all boxes

    boxes = [
        (0.03, BOX_Y, 0.15, BOX_H, "#E3F2FD", "CMAPSS\nSensor Data\n(21 sensors,\nmulti-cycle)"),
        (0.23, BOX_Y, 0.15, BOX_H, "#FFF3E0", "Preprocessing\n& Feature\nSelection\n(14 sensors,\nnormalised)"),
        (0.43, BOX_Y, 0.15, BOX_H, "#E8F5E9", "Prompt\nConstruction\n(zero-shot /\nfew-shot\ntemplates)"),
        (0.63, BOX_Y, 0.15, BOX_H, "#F3E5F5", "Llama 3.1 8B\n(local inference\nvia Ollama,\nno fine-tuning)"),
        (0.83, BOX_Y, 0.15, BOX_H, "#FCE4EC", "Output\nRUL estimate\n+ Natural\nlanguage\nreasoning"),
    ]

    for (x, y, w, h, color, label) in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#555", linewidth=1.5,
            transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                transform=ax.transAxes, multialignment="center")

    # Main horizontal arrows BETWEEN boxes
    arrow_xs = [(0.18, 0.23), (0.38, 0.43), (0.58, 0.63), (0.78, 0.83)]
    labels_a = ["Select 14 sensors", "Serialise to text",
                "API call (local)",   "Parse response"]
    for (x1, x2), lbl in zip(arrow_xs, labels_a):
        # main arrow
        ax.annotate("", xy=(x2, ARROW_Y), xytext=(x1, ARROW_Y),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2))
        xc = (x1 + x2) / 2
        # vertical guide from label down to the arrow
        ax.plot([xc, xc], [ARROW_Y + 0.02, LABEL_Y - 0.04],
                color="#999", lw=0.8, ls=":", transform=ax.transAxes,
                clip_on=False)
        # label with white box for clarity
        ax.text(xc, LABEL_Y, lbl, ha="center", va="center",
                fontsize=9.5, color="#222", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#bbb", lw=0.8))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_fig(fig, "fig1_framework")
    print("Figure 1 saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 2 — Bar chart: RMSE / MAE comparison (FD001, n=5)
# ════════════════════════════════════════════════════════════
def fig_bar_comparison(results_dict):
    models = list(results_dict.keys())
    rmses  = [results_dict[m]["RMSE"]  for m in models]
    maes   = [results_dict[m]["MAE"]   for m in models]
    colors = [results_dict[m]["color"] for m in models]

    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, vals, ylabel, title in zip(
            axes,
            [rmses, maes],
            ["RMSE (cycles)", "MAE (cycles)"],
            ["Root Mean Square Error (RMSE)", "Mean Absolute Error (MAE)"]):
        bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="white",
                      linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("RUL Prediction Performance — NASA CMAPSS FD001 (100 test engines, n=5)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_bar_comparison")
    print("Figure 2 saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 3 — Scatter: predicted vs true RUL (LLM only)
# ════════════════════════════════════════════════════════════
def fig_scatter(llm_preds_dict, results_dict):
    models_plot = list(llm_preds_dict.keys())
    n = len(models_plot)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    lim = [0, 130]
    for ax, model in zip(axes, models_plot):
        t, p = llm_preds_dict[model]
        rmse = results_dict[model]["RMSE"]
        color = results_dict[model]["color"]
        ax.scatter(t, p, alpha=0.65, s=45, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction", zorder=2)
        ax.fill_between(lim, [v - 20 for v in lim], [v + 20 for v in lim],
                        alpha=0.08, color="grey", label="±20 cycles band")
        ax.set_xlabel("True RUL (cycles)")
        label_clean = model.replace("\n", " ")
        ax.set_title(f"{label_clean}\nRMSE = {rmse:.1f}", fontweight="bold")
        ax.set_xlim(0, 130); ax.set_ylim(0, 130)
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Predicted RUL (cycles)")
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Predicted vs. True Remaining Useful Life — Llama 3.1 8B, NASA CMAPSS FD001 (100 engines, n=5)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_scatter")
    print("Figure 3 saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 3b — Scatter: predicted vs true RUL at n=30 window
# ════════════════════════════════════════════════════════════
def fig_scatter_n30():
    """Same scatter layout as Fig 3 but for the n=30 prompt-window traces.
    Shows the categorical cluster shift toward mid-life values (85-105 cycles)."""
    panels = [
        ("Llama Zero-Shot",       "traces_FD001_zero_shot_k0_n30",  C_LLM),
        ("Llama Few-Shot k=3",    "traces_FD001_few_shot_k3_n30",   "#8BC34A"),
        ("Llama Few-Shot k=5",    "traces_FD001_few_shot_k5_n30",   "#7CB342"),
        ("Llama Few-Shot k=10",   "traces_FD001_few_shot_k10_n30",  "#33691E"),
    ]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]
    lim = [0, 130]
    for ax, (label, fname, color) in zip(axes, panels):
        path = RESULTS / f"{fname}.json"
        if not path.exists():
            ax.text(0.5, 0.5, f"missing:\n{fname}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="red")
            ax.set_title(label, fontweight="bold")
            continue
        t, p = load_traces(path)
        rmse = metric_rmse(t, p)
        ax.scatter(t, p, alpha=0.65, s=45, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot(lim, lim, "k--", lw=1.2, label="Perfect prediction", zorder=2)
        ax.fill_between(lim, [v - 20 for v in lim], [v + 20 for v in lim],
                        alpha=0.08, color="grey", label="±20 cycles band")
        ax.set_xlabel("True RUL (cycles)")
        ax.set_title(f"{label}\nRMSE = {rmse:.1f}", fontweight="bold")
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
    print("Figure 3b saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 3c — Distinct values + top-1 share per configuration
# ════════════════════════════════════════════════════════════
def fig_distinct_values():
    """Two-panel bar chart documenting categorical-collapse behaviour:
    (left) number of distinct predicted values across the 100 test engines;
    (right) share of predictions assigned to the single most frequent value."""
    configs = [
        ("ZS", "zero_shot", 0),
        ("k=3", "few_shot",  3),
        ("k=5", "few_shot",  5),
        ("k=10", "few_shot", 10),
    ]
    windows = [5, 15, 30]
    win_colors = {5: "#A5D6A7", 15: "#4CAF50", 30: "#1B5E20"}

    labels, distinct, top1_share, top1_val, colors = [], [], [], [], []
    for cfg_label, mode, k in configs:
        for n in windows:
            suffix = f"_n{n}" if n != 5 else ""
            fname = (f"traces_FD001_zero_shot_k0{suffix}.json" if mode == "zero_shot"
                     else f"traces_FD001_few_shot_k{k}{suffix}.json")
            path = RESULTS / fname
            if not path.exists():
                continue
            _, preds = load_traces(path)
            preds = np.array(preds, dtype=int)
            uniq, counts = np.unique(preds, return_counts=True)
            top_idx = int(np.argmax(counts))
            labels.append(f"{cfg_label}\nn={n}")
            distinct.append(int(len(uniq)))
            top1_share.append(int(counts[top_idx]))   # out of 100 engines
            top1_val.append(int(uniq[top_idx]))
            colors.append(win_colors[n])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # LEFT panel: distinct values
    bars = axes[0].bar(x, distinct, color=colors, edgecolor="white",
                       linewidth=0.8, zorder=3)
    for bar, v in zip(bars, distinct):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     str(v), ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Number of distinct predicted values\n(out of 100 engines)")
    axes[0].set_title("Categorical-collapse: output cardinality", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, max(distinct) + 3)

    # RIGHT panel: top-1 share
    bars = axes[1].bar(x, top1_share, color=colors, edgecolor="white",
                       linewidth=0.8, zorder=3)
    for bar, v, val in zip(bars, top1_share, top1_val):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f"{v}%\n(={val})", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Share assigned to the most frequent value (%)")
    axes[1].set_title("Top-1 concentration (label = anchor value)", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 100)

    # legend (window colors) — manual
    handles = [mpatches.Patch(color=win_colors[n], label=f"n={n}") for n in windows]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Predicted-Value Distribution — Llama 3.1 8B, NASA CMAPSS FD001",
                 fontweight="bold", y=1.07)
    fig.tight_layout()
    save_fig(fig, "fig3c_distinct_values")
    print("Figure 3c saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 4 — Distribution: predicted vs true (FD001 ZS, 100 eng)
# ════════════════════════════════════════════════════════════
def fig_distribution(true_zs, pred_zs):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.arange(0, 131, 10)
    ax.hist(true_zs, bins=bins, alpha=0.55, color=C_RF,
            label="True RUL distribution", edgecolor="white", zorder=3)
    ax.hist(pred_zs, bins=bins, alpha=0.55, color=C_LLM,
            label="Llama 3.1 8B predictions (zero-shot)", edgecolor="white", zorder=3)

    ax.axvline(np.mean(true_zs), color=C_RF,   lw=2, ls="--",
               label=f"Mean true RUL = {np.mean(true_zs):.1f}")
    ax.axvline(np.mean(pred_zs), color=C_LLM,  lw=2, ls="--",
               label=f"Mean predicted RUL = {np.mean(pred_zs):.1f}")

    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Number of engines")
    ax.set_title("Distribution of True vs. Predicted RUL — Llama 3.1 8B Zero-Shot, FD001 (100 engines, n=5)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "fig4_distribution")
    print("Figure 4 saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 5 — Few-shot ablation curves: RMSE vs k for n in {5,15,30}
# ════════════════════════════════════════════════════════════
def _trace_filename(dataset, mode, k, n):
    suffix = f"_n{n}" if n != 5 else ""
    if mode == "zero_shot":
        return f"traces_{dataset}_zero_shot_k0{suffix}.json"
    return f"traces_{dataset}_few_shot_k{k}{suffix}.json"


def fig_fewshot_ablation():
    """Multi-curve RMSE vs k, one curve per prompt window n ∈ {5, 15, 30}."""
    ks = [0, 3, 5, 10]
    windows = [5, 15, 30]
    colors_n = {5: "#A5D6A7", 15: "#4CAF50", 30: "#1B5E20"}
    markers   = {5: "o", 15: "s", 30: "D"}

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Variance band — built from the 4 ZS FD001 n=30 replicates
    var_runs = []
    for suffix in ["_n30", "_n30_run2", "_n30_run3", "_n30_run4"]:
        p = RESULTS / f"traces_FD001_zero_shot_k0{suffix}.json"
        if p.exists():
            t_, p_ = load_traces(p)
            var_runs.append(metric_rmse(t_, p_))
    if len(var_runs) >= 2:
        mu, sd = np.mean(var_runs), np.std(var_runs, ddof=1)
        ax.errorbar([0], [mu], yerr=[sd], fmt="none",
                    ecolor="#1B5E20", elinewidth=1.6, capsize=5, zorder=2,
                    label=f"ZS n=30 ±1 SD ({sd:.2f}) over {len(var_runs)} runs")

    for n in windows:
        rmses = []
        for k in ks:
            mode = "zero_shot" if k == 0 else "few_shot"
            p = RESULTS / _trace_filename("FD001", mode, k, n)
            if p.exists():
                t_, pr_ = load_traces(p)
                rmses.append(metric_rmse(t_, pr_))
            else:
                rmses.append(np.nan)
        ax.plot(ks, rmses, "-", color=colors_n[n],
                marker=markers[n], ms=8, lw=2.2, zorder=3,
                label=f"Prompt window n={n}")
        for k, r in zip(ks, rmses):
            if not np.isnan(r):
                ax.annotate(f"{r:.1f}", (k, r), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5,
                            color=colors_n[n])

    # Baselines as horizontal reference (LSTM in range; RF off-scale, in legend)
    ax.axhline(LSTM_RMSE_FD001, color=C_LSTM, lw=1.6, ls="--",
               label=f"LSTM baseline (RMSE = {LSTM_RMSE_FD001})")

    ax.set_xlabel("Number of few-shot examples (k)")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("Few-Shot Ablation by Prompt Window — Llama 3.1 8B, NASA CMAPSS FD001",
                 fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlim(-1, 12)
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.text(0.99, 0.97, f"Random Forest baseline RMSE = {RF_RMSE_FD001} (off-scale)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    fig.tight_layout()
    save_fig(fig, "fig5_fewshot_curve")
    print("Figure 5 saved.")


# ════════════════════════════════════════════════════════════
# FIGURE 6 — NEW: NASA Score vs prompt window (U-shape trade-off)
# ════════════════════════════════════════════════════════════
def fig_nasa_ushape():
    windows = [5, 15, 30]
    configs = [
        ("FD001", "zero_shot", 0, "FD001 ZS",          "#90CAF9", "o"),
        ("FD001", "few_shot",  3, "FD001 k=3",         "#42A5F5", "s"),
        ("FD001", "few_shot",  5, "FD001 k=5",         "#1E88E5", "D"),
        ("FD001", "few_shot", 10, "FD001 k=10",        "#0D47A1", "^"),
        ("FD003", "zero_shot", 0, "FD003 ZS",          "#FFCC80", "o"),
        ("FD003", "few_shot",  3, "FD003 k=3",         "#FFA726", "s"),
        ("FD003", "few_shot",  5, "FD003 k=5",         "#FB8C00", "D"),
        ("FD003", "few_shot", 10, "FD003 k=10",        "#E65100", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, dataset in zip(axes, ["FD001", "FD003"]):
        for ds, mode, k, label, color, marker in configs:
            if ds != dataset:
                continue
            ys = []
            for n in windows:
                p = RESULTS / _trace_filename(ds, mode, k, n)
                if p.exists():
                    t_, pr_ = load_traces(p)
                    ys.append(metric_nasa(t_, pr_))
                else:
                    ys.append(np.nan)
            ax.plot(windows, ys, "-", color=color, marker=marker, ms=8,
                    lw=2.0, label=label, zorder=3)

        ax.set_xlabel("Prompt history window n (cycles)")
        if dataset == "FD001":
            ax.set_ylabel("NASA Score (lower is better)")
        ax.set_title(f"NASA CMAPSS {dataset}", fontweight="bold")
        ax.set_xticks(windows)
        ax.set_yscale("log")
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95)

    fig.suptitle("NASA Score vs. Prompt History Window — U-shape Trade-off (log scale)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_nasa_ushape")
    print("Figure 6 saved.")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    # ── Load FD001 n=5 traces for figs 2/3/4 ─────────────────
    zs_true,  zs_pred  = load_traces(RESULTS / "traces_FD001_zero_shot_k0.json")
    fs3_true, fs3_pred = load_traces(RESULTS / "traces_FD001_few_shot_k3.json")

    results_dict = {
        "Random\nForest":      {"RMSE": RF_RMSE_FD001,   "MAE": RF_MAE_FD001,   "color": C_RF},
        "LSTM":                 {"RMSE": LSTM_RMSE_FD001, "MAE": LSTM_MAE_FD001, "color": C_LSTM},
        "Llama\nZero-Shot":     {"RMSE": metric_rmse(zs_true, zs_pred),
                                 "MAE":  metric_mae(zs_true,  zs_pred), "color": C_LLM},
        "Llama\nFew-Shot k=3": {"RMSE": metric_rmse(fs3_true, fs3_pred),
                                 "MAE":  metric_mae(fs3_true, fs3_pred), "color": "#8BC34A"},
    }

    llm_scatter_preds = {
        "Llama\nZero-Shot":    (zs_true, zs_pred),
        "Llama\nFew-Shot k=3": (fs3_true, fs3_pred),
    }

    k_colors = {5: "#7CB342", 10: "#33691E"}
    for k in [5, 10]:
        p = RESULTS / f"traces_FD001_few_shot_k{k}.json"
        if p.exists():
            t_, p_ = load_traces(p)
            label = f"Llama\nFew-Shot k={k}"
            results_dict[label] = {
                "RMSE": metric_rmse(t_, p_),
                "MAE":  metric_mae(t_, p_),
                "color": k_colors[k],
            }
            llm_scatter_preds[label] = (t_, p_)

    # ── Generate figures ─────────────────────────────────────
    fig_framework()
    fig_bar_comparison(results_dict)
    fig_scatter(llm_scatter_preds, results_dict)
    fig_scatter_n30()
    fig_distribution(zs_true, zs_pred)
    fig_fewshot_ablation()
    fig_nasa_ushape()
    fig_distinct_values()

    print(f"\nAll figures saved to: {FIGS.absolute()}")
    print("Formats: PDF (300 DPI) + PNG (150 DPI). PDF is the print-quality version.")


if __name__ == "__main__":
    main()
