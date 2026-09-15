"""
P3.9 - A worked example for the trend-reversal result of Section 4.5, proposed but
not yet approved for insertion into the manuscript.

Section 4.5 reports that reversing every sensor trend in the window leaves the
model's stated direction unchanged for s11 in every pair where the window
actually reversed. This picks ONE such pair to show rather than only state, by a
rule fixed before looking at any case: the lowest-numbered engine among the
(engine, sensor) pairs for s11 where (a) the window's real trend reversed between
the repl and revdata arms and (b) the model's claimed direction did not change.
No case was read before this rule was applied.

That rule selects FD001 engine 1. Both facts below are read straight from
results_p0/traces_p0_FD001_zs_n30_{repl,revdata}.json and
results_p0/p0_23_claims.csv, not chosen for effect:
  - the actual window: falls net -0.16 in the original order, rises net +0.16
    reversed - the same 30 numbers, reordered;
  - the model's claim: "increase" in both arms, and its predicted RUL moves by
    one cycle (42 to 43) despite the trend it describes having flipped sign.

Output: figures_png/fig7.png, at the same 2400 px / 15.5 cm print width as every
other figure. Run this AFTER p1/export_figures_png.py, which does not know about
this figure and would not overwrite it, but do not rely on that - run it last.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402

OUT = EXP / "figures_png" / "fig7.png"
SENSOR = "s11"
# matches the 2400 px / 15.5 cm print width every other figure is normalised to
# by p1/export_figures_png.py (TARGET_W); found empirically since bbox_inches="tight"
# does not preserve figsize*dpi exactly
TARGET_W = 2400
SAVE_DPI = 330

INK, ACCENT, GRID, GREY = "#1F3B57", "#C2553F", "#D9D9D9", "#8C8C8C"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                     "axes.edgecolor": GREY, "axes.linewidth": 0.8})


def pick_engine():
    """The selection rule, run against the data rather than hardcoded."""
    a = pd.read_csv(EXP / "results_p0" / "p0_23_claims.csv")
    s = a[a.sensor == SENSOR]
    piv = s.pivot_table(index="engine_id", columns="arm",
                        values=["claimed", "actual_shown"], aggfunc="first")
    piv.columns = [f"{x}_{y}" for x, y in piv.columns]
    piv = piv.dropna(subset=["claimed_repl", "claimed_revdata",
                             "actual_shown_repl", "actual_shown_revdata"])
    dirn = ["increase", "decrease"]
    both = piv[piv.claimed_repl.isin(dirn) & piv.claimed_revdata.isin(dirn)]
    reversed_ = both[both.actual_shown_repl != both.actual_shown_revdata]
    unchanged = reversed_[reversed_.claimed_repl == reversed_.claimed_revdata]
    print(f"{SENSOR}: {len(both)} pairs with a claim in both arms, "
          f"{len(reversed_)} with the window reversed, "
          f"{len(unchanged)} of those left the claim unchanged "
          f"({100 * len(unchanged) / max(len(reversed_), 1):.0f}%).")
    eid = int(unchanged.index.min())
    row = unchanged.loc[eid]
    return eid, row.claimed_repl, len(unchanged), len(reversed_)


def main():
    eid, claim, n_unchanged, n_reversed = pick_engine()

    cfg.N_CYCLES_IN_PROMPT = 30
    _, test_df, _ = ex.load_cmapss("FD001")
    window = test_df[test_df.engine_id == eid][SENSOR].tail(30).to_numpy()
    reversed_window = window[::-1]

    traces = {arm: json.loads((EXP / "results_p0" /
                               f"traces_p0_FD001_zs_n30_{arm}.json").read_text(encoding="utf-8"))
             for arm in ("repl", "revdata")}
    resp = {arm: next(t for t in traces[arm] if t["engine_id"] == eid) for arm in traces}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    x = np.arange(1, 31)
    ymin, ymax = min(window.min(), reversed_window.min()), max(window.max(), reversed_window.max())
    pad = 0.14 * (ymax - ymin)
    panels = [("Original window (repl arm)", window, resp["repl"]),
             ("Time-reversed window (revdata arm)", reversed_window, resp["revdata"])]
    for ax, (title, series, tr) in zip(axes, panels):
        ax.plot(x, series, color=INK, lw=1.3, marker="o", ms=2.8, zorder=3, alpha=0.75)
        net = series[-1] - series[0]
        # the arrow is literally the quantity the extractor scores: sign(last - first)
        ax.annotate("", xy=(30, series[-1]), xytext=(1, series[0]),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2,
                                    mutation_scale=16, shrinkA=0, shrinkB=0), zorder=5)
        ax.plot([1, 30], [series[0], series[-1]], "o", color=ACCENT, ms=6, zorder=6)
        ax.set_title(title, fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("Cycle in the prompt window (oldest to most recent)", fontsize=8.5)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(-1, 32)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.97, 0.94, f"first-to-last: {net:+.2f}", transform=ax.transAxes,
                fontsize=8.5, color=ACCENT, ha="right", va="top", weight="bold")
        ax.text(0.03, 0.06, f"model's claim: “{claim}”\npredicted RUL: {tr['pred_rul']}",
                transform=ax.transAxes, fontsize=9, color=ACCENT, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ACCENT, lw=0.8))
    axes[0].set_ylabel(f"{SENSOR} (min-max normalised)", fontsize=9)
    fig.suptitle(f"FD001 engine {eid}, sensor {SENSOR}: same claim, opposite trend",
                fontsize=12.5, color=INK, y=1.10)
    # a one-line key only, to read the plot on its own; the method belongs in the
    # caption paragraph below the figure, in the same place as every other caption
    fig.text(0.5, 1.015, "— value shown to the model        "
             "→ first-to-last direction (scored in Section 3.5)",
             ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    from PIL import Image
    im = Image.open(OUT)
    if im.width != TARGET_W:
        im = im.resize((TARGET_W, round(im.height * TARGET_W / im.width)), Image.LANCZOS)
        im.save(OUT)
    print(f"\n[saved] {OUT} ({Image.open(OUT).size[0]}x{Image.open(OUT).size[1]} px)")

    caption = (
        f"Figure X. A single (engine, sensor) pair from the trend-reversal test of Figure 6, "
        f"chosen before it was read: the lowest-numbered FD001 engine among the pairs where "
        f"{SENSOR}'s claim did not change although the window's first-to-last direction did "
        f"({n_unchanged} of {n_reversed} such pairs for this sensor). The model states "
        f"“{claim}” for {SENSOR} in both the original window (first-to-last -0.16) "
        f"and the same window reversed in time (first-to-last +0.16); its predicted RUL moves "
        f"by one cycle, {resp['repl']['pred_rul']} to {resp['revdata']['pred_rul']}.")
    print(f"\nProposed caption, unnumbered pending approval:\n{caption}")

    print(f"\nrepl reasoning:\n  {resp['repl']['reasoning']}")
    print(f"\nrevdata reasoning:\n  {resp['revdata']['reasoning']}")


if __name__ == "__main__":
    main()
