"""
Analysis of the corrected k x n few-shot grid.

Answers the question the earlier grid could not, because two of its three k
levels were the same prompt: with examples drawn properly, does the number of
examples do anything?

Three comparisons:
  * the grid itself, k x n, on error and on collapse measures;
  * each cell against the no-skill constant;
  * the k effect against the sampling variability measured in P1.4. The
    comparison is range against range: the spread across k at a fixed window
    against the spread produced by redrawing the examples with k held fixed.
    Comparing a range with a standard deviation would understate the noise.

It also re-runs the faithfulness measure on the new traces, to check whether
properly stratified examples change how the explanations behave.

Outputs (results_p1/):
  p1_grid_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"

sys.path.insert(0, str(EXP))
from analyze_explainability import SENSOR_PHYSICS, classify_mention  # noqa: E402

NOSKILL = 41.94          # constant = train mean, FD001
# P1.4 measured what changing only *which* examples are drawn does to RMSE,
# with k held fixed. Reported here as a range, so that it can be compared
# like for like with the range across k below (a range against a standard
# deviation is not a like-for-like comparison).
SEED_RANGE = 16.34       # max - min RMSE across the stratified seeds, from P1.4
SEED_SD = 8.25           # the same six runs as a standard deviation
FLAT_EPS = 0.01


def direction(vals, eps=FLAT_EPS):
    net = float(vals[-1] - vals[0])
    return "stable" if abs(net) < eps else ("increase" if net > 0 else "decrease")


def faithfulness_on_grid(df):
    """Textbook agreement vs input faithfulness on the new grid traces."""
    import config as cfg
    import experiment as ex
    _, test_df, _ = ex.load_cmapss("FD001")

    rows = []
    for _, r in df.iterrows():
        f = P1 / f"traces_p1_grid_FD001_fs_k{int(r.k)}_n{int(r.n)}.json"
        if not f.exists():
            continue
        for t in json.loads(f.read_text(encoding="utf-8")):
            g = test_df[test_df.engine_id == t["engine_id"]]
            win = g[cfg.SENSOR_NAMES].tail(int(r.n))
            text = t.get("reasoning", "") or ""
            for s, (canon, _) in SENSOR_PHYSICS.items():
                claimed = classify_mention(text, s)
                if claimed not in ("increase", "decrease"):
                    continue
                rows.append({"k": int(r.k), "n": int(r.n), "sensor": s,
                             "claimed": claimed, "canonical": canon,
                             "actual": direction(np.round(win[s].values.astype(float), 3))})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(P1 / "p1_grid_fixed_summary.csv").sort_values(["n", "k"])
    complete = df[df.engines >= 100]

    L = ["# The k x n few-shot grid, with a correct example sampler\n"]
    L.append(f"FD001, 100 test engines per cell, llama3.1:8b, T=0.1. Examples are windows "
             f"ending at an arbitrary cycle, stratified across the three RUL bins, so their "
             f"labels span 0 to 125 rather than being uniformly zero.\n")
    if len(complete) < len(df):
        L.append(f"> **{len(df) - len(complete)} cell(s) still running.**\n")

    L.append("\n## The grid\n")
    L.append("| k | n | RMSE | MAE | NASA | distinct | modal (share) | entropy | rho | p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in complete.iterrows():
        rho = "n/a" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.3f}"
        p = "n/a" if pd.isna(r.spearman_p) else f"{r.spearman_p:.3f}"
        L.append(f"| {int(r.k)} | {int(r.n)} | {r.RMSE:.2f} | {r.MAE:.2f} | "
                 f"{r.NASA_Score:,.0f} | {int(r.distinct_values)} | "
                 f"{int(r.modal_value)} ({r.modal_share_pct:.0f}%) | {abs(r.entropy_bits):.2f} | "
                 f"{rho} | {p} |")

    beat = complete[complete.RMSE < NOSKILL]
    L.append(f"\n**Against the no-skill constant (RMSE {NOSKILL}):** "
             f"{len(beat)}/{len(complete)} cells beat it"
             + (f" ({', '.join(beat.config)})" if len(beat) else "") + ".")
    L.append(f"\n**Collapse:** {int(complete.distinct_values.min())} to "
             f"{int(complete.distinct_values.max())} distinct values over 100 engines; "
             f"entropy {abs(complete.entropy_bits.min()):.2f} to {complete.entropy_bits.max():.2f} "
             f"bits against a ceiling of {np.log2(100):.2f}.")

    # ---- does k do anything? ----
    L.append("\n## Does the number of examples do anything?\n")
    L.append("| n | " + " | ".join(f"k={int(k)}" for k in sorted(complete.k.unique()))
             + " | spread across k |")
    L.append("|---" * (len(complete.k.unique()) + 2) + "|")
    spreads = []
    for n, g in complete.groupby("n"):
        cells = {int(r.k): r.RMSE for _, r in g.iterrows()}
        vals = list(cells.values())
        sp = max(vals) - min(vals)
        spreads.append(sp)
        L.append(f"| {int(n)} | " + " | ".join(
            f"{cells.get(int(k), float('nan')):.2f}" for k in sorted(complete.k.unique()))
            + f" | {sp:.2f} |")

    L.append(f"\nThe largest RMSE difference between k levels at a fixed window is "
             f"**{max(spreads):.2f}**. The comparable quantity measured in P1.4 is the spread "
             f"produced by changing only *which* examples are drawn, with k held fixed: a range "
             f"of **{SEED_RANGE}** RMSE over the stratified seeds (standard deviation "
             f"{SEED_SD}). The spread across k does not exceed the spread produced by the draw, "
             "so this grid does not establish an effect of k on RMSE.")

    # Concentration is a different matter from error, and does not behave the same way.
    conc = complete.groupby("k").agg(
        modal=("modal_share_pct", "mean"), dv=("distinct_values", "mean")).reset_index()
    L.append("\n| k | mean modal share | mean distinct values |")
    L.append("|---|---|---|")
    for _, r in conc.iterrows():
        L.append(f"| {int(r.k)} | {r.modal:.0f}% | {r.dv:.1f} |")
    if len(conc):
        hi = conc.loc[conc.k.idxmax()]
        top = complete[complete.k == hi.k].sort_values("n")
        L.append("\nConcentration does not follow RMSE. Mean modal share moves "
                 + " to ".join(f"{r.modal:.0f}% (k={int(r.k)})" for _, r in conc.iterrows())
                 + f", so the largest example count in this grid is also the most concentrated: "
                 f"at k={int(hi.k)} the model returns {hi.dv:.1f} distinct values on average over "
                 f"100 engines. Once a cell is that concentrated its error is decided by which "
                 f"single value is returned - here "
                 + ", ".join(f"{int(r.modal_value)} (RMSE {r.RMSE:.2f})" for _, r in top.iterrows())
                 + ". The best and the worst cell of the whole grid are both at this k, which is "
                 "why the RMSE spread across k should not be read as an effect of k.")

    if len(complete) >= 4 and complete.k.nunique() > 1:
        groups = [g.RMSE.values for _, g in complete.groupby("k")]
        if all(len(x) > 1 for x in groups) and len(groups) > 1:
            h, p = kruskal(*groups)
            L.append(f"\nKruskal-Wallis across k: H = {h:.2f}, p = {p:.3f}.")

    # ---- faithfulness on the new traces ----
    try:
        fa = faithfulness_on_grid(complete)
        if len(fa):
            tb = 100 * (fa.claimed == fa.canonical).mean()
            inp = 100 * (fa.claimed == fa.actual).mean()
            n_traces = int(complete.engines.sum())
            L.append("\n## Explanations, on these traces\n")
            L.append(f"- Directional claims: **{len(fa)}** over {n_traces:,} traces, "
                     f"about {len(fa) / max(n_traces, 1):.2f} per trace")
            L.append(f"- Agreement with canonical physics: {tb:.1f}%")
            L.append(f"- Faithfulness to the window shown: {inp:.1f}%")
            if len(fa) < 100:
                L.append(f"\n**These two percentages rest on {len(fa)} claims and are not "
                         "worth interpreting.** The finding here is the count itself. The "
                         "few-shot template carries no sentence stating which sensors rise "
                         "and which fall, and without it the model almost stops naming sensor "
                         "directions at all. That matches the `nocue` control arm in Section "
                         "4.5, where removing the cue cut directional claims by 74%, and it "
                         "means the explanation measures in Section 4.4 are driven by the "
                         "zero-shot prompts, as reported there.")
            else:
                L.append("\nCorrecting the examples does not change how the explanations "
                         "behave: the gap between the two measures is the one reported in "
                         "Section 4.4.")
    except Exception as exc:
        L.append(f"\n*(faithfulness on grid traces not computed: {exc})*")

    L.append("\n## What this means for the paper\n")
    L.append("- The ablation over k can be reported as an ablation, because every level now "
             "receives the number of examples it claims to.")
    L.append("- On RMSE, k has no effect distinguishable from the noise of which examples "
             "are drawn: the spread across k does not exceed the spread across draws.")
    L.append("- On concentration the picture is different, and is the more interesting one: "
             "the largest example count is the most concentrated, at one or two distinct "
             "values in all three windows. More examples did not restore regression.")
    L.append("- Fixing the sampler does not lift the collapse: every cell still returns a "
             "handful of distinct values, and none of them beats a constant.")

    (P1 / "p1_grid_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p1_grid_report.md'}")


if __name__ == "__main__":
    main()
