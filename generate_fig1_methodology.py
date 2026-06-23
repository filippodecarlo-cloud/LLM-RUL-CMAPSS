"""
generate_fig1_methodology.py
Generates the NEW Figure 1: a step-by-step methodology flowchart for paper v7.

Five stages:
  1. Data (NASA CMAPSS)
  2. Preprocessing
  3. Modelling — two parallel branches: Baselines (RF, LSTM) and LLM pipeline
  4. Quantitative evaluation (RMSE / MAE / NASA Score)
  5. Two diagnostic analyses (output distribution + explainability)

Replaces the old fig1_framework (LLM-only pipeline). Saves PNG (300 dpi) + PDF
in experiment/figures/, consistent with the other figures.

Run:  python generate_fig1_methodology.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palettes: (face, edge, text) ─────────────────────────────────────
GRAY   = ("#ECEFF1", "#607D8B", "#263238")
BLUE   = ("#E3F2FD", "#1976D2", "#0D47A1")   # baselines
PURPLE = ("#F3E5F5", "#8E24AA", "#4A148C")   # LLM
TEAL   = ("#E0F2F1", "#00897B", "#004D40")   # diagnostic analyses
ARROW  = "#78909C"


def box(ax, cx, cy, w, h, title, subtitle, palette, title_fs=11, sub_fs=8.5):
    face, edge, text = palette
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.8",
        linewidth=1.4, facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(p)
    if subtitle:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=text, zorder=3)
        ax.text(cx, cy - h * 0.26, subtitle, ha="center", va="center",
                fontsize=sub_fs, color=text, alpha=0.82, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=text, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=1)


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=ARROW, lw=1.6, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 98)
    ax.axis("off")

    # Stage 1 — Data
    box(ax, 50, 92, 54, 8, "NASA CMAPSS data",
        "FD001 & FD003  ·  100 test engines each", GRAY)
    arrow(ax, 50, 88, 50, 85)

    # Stage 2 — Preprocessing
    box(ax, 50, 81, 54, 8, "Preprocessing",
        "14 sensors  ·  min-max norm  ·  RUL cap 125", GRAY)

    # Split to branches
    line(ax, 50, 77, 50, 73)
    line(ax, 26.5, 73, 73.5, 73)
    arrow(ax, 26.5, 73, 26.5, 70.5)
    arrow(ax, 73.5, 73, 73.5, 70.5)

    # ── Left branch: Baselines ──────────────────────────────────────────────
    ax.add_patch(Rectangle((7, 47), 39, 22, fill=False, linestyle=(0, (4, 3)),
                           edgecolor="#B0BEC5", linewidth=1.0, zorder=0))
    ax.text(9, 67, "BASELINES", fontsize=9, fontweight="bold",
            color="#546E7A", zorder=3)
    box(ax, 26.5, 62, 36, 7.5, "Random Forest",
        "200 trees · 30-cycle window · 42 feat.", BLUE, title_fs=10.5, sub_fs=8)
    box(ax, 26.5, 52.5, 36, 7.5, "LSTM",
        "64–32 units · dropout 0.2", BLUE, title_fs=10.5, sub_fs=8)

    # ── Right branch: LLM pipeline ──────────────────────────────────────────
    ax.add_patch(Rectangle((54, 39), 39, 30, fill=False, linestyle=(0, (4, 3)),
                           edgecolor="#B0BEC5", linewidth=1.0, zorder=0))
    ax.text(56, 67, "LLM PIPELINE", fontsize=9, fontweight="bold",
            color="#546E7A", zorder=3)
    box(ax, 73.5, 62, 36, 7, "Prompt construction",
        "zero / few-shot · n=5,15,30 · k=3,5,10", PURPLE, title_fs=10.5, sub_fs=7.5)
    arrow(ax, 73.5, 58.5, 73.5, 56.5)
    box(ax, 73.5, 53, 36, 7, "Local inference",
        "Llama 3.1 8B · Ollama · T=0.1", PURPLE, title_fs=10.5, sub_fs=7.5)
    arrow(ax, 73.5, 49.5, 73.5, 47.5)
    box(ax, 73.5, 44, 36, 7, "Parse response",
        "RUL_ESTIMATE + REASONING", PURPLE, title_fs=10.5, sub_fs=7.5)

    # Convergence to evaluation
    line(ax, 26.5, 48.25, 26.5, 31)
    line(ax, 73.5, 40.5, 73.5, 31)
    line(ax, 26.5, 31, 73.5, 31)
    arrow(ax, 50, 31, 50, 28)

    # Stage 4 — Quantitative evaluation
    box(ax, 50, 24, 54, 8, "Quantitative evaluation",
        "RMSE  ·  MAE  ·  NASA Score", GRAY)

    # Split to diagnostic analyses
    line(ax, 50, 20, 50, 16.5)
    line(ax, 28, 16.5, 72, 16.5)
    arrow(ax, 28, 16.5, 28, 14)
    arrow(ax, 72, 16.5, 72, 14)

    # Stage 5 — Diagnostic analyses
    box(ax, 28, 10, 40, 8, "Output distribution",
        "distinct values · top-1 share", TEAL, title_fs=10.5, sub_fs=8)
    box(ax, 72, 10, 40, 8, "Explainability analysis",
        "2,700 traces · sensor directions", TEAL, title_fs=10.5, sub_fs=8)

    fig.tight_layout(pad=0.5)
    png = os.path.join(OUT_DIR, "fig1_methodology.png")
    pdf = os.path.join(OUT_DIR, "fig1_methodology.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved:\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
