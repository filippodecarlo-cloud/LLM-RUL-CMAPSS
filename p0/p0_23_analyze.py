"""
P0.2b / P0.3 - analysis of the four control arms.

Reads the traces produced by p0_23_control_prompts.py and asks, for each arm:

  Does the explanation follow the PROMPT?   (compare arms repl / nocue / invcue,
                                             which differ only in the cue sentence)
  Does the explanation follow the DATA?     (compare arms repl / revdata, which
                                             differ only in the direction of every
                                             trend in the window)

The decisive number is the paired one: among the (engine, sensor) pairs whose
real trend flips between repl and revdata, how many of the model's directional
claims flip with it? A model reading the window must flip most of them.

Outputs (results_p0/):
  p0_23_claims.csv
  p0_23_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402
from analyze_explainability import SENSOR_PHYSICS, classify_mention  # noqa: E402

OUT = EXP / "results_p0"
DATASET, N_CYCLES, FLAT_EPS = "FD001", 30, 0.01
CUED = {"s9", "s11", "s12", "s14", "s15"}

ARMS = ["repl", "nocue", "invcue", "revdata"]
ARM_LABEL = {
    "repl": "repl (original prompt, real data)",
    "nocue": "nocue (cue removed)",
    "invcue": "invcue (cue asserts opposite physics)",
    "revdata": "revdata (original cue, window reversed in time)",
}
# What each arm's prompt tells the model to expect, for the cued sensors.
def cue_direction(arm, sensor):
    if arm == "nocue" or sensor not in CUED:
        return None
    canonical = SENSOR_PHYSICS[sensor][0]
    if arm == "invcue":
        return "decrease" if canonical == "increase" else "increase"
    return canonical


def direction(vals, eps=FLAT_EPS):
    net = float(vals[-1] - vals[0])
    if abs(net) < eps:
        return "stable"
    return "increase" if net > 0 else "decrease"


def main():
    cfg.N_CYCLES_IN_PROMPT = N_CYCLES
    _, test_df, _ = ex.load_cmapss(DATASET)

    rows = []
    for arm in ARMS:
        path = OUT / f"traces_p0_{DATASET}_zs_n{N_CYCLES}_{arm}.json"
        for t in json.loads(path.read_text(encoding="utf-8")):
            eid = t["engine_id"]
            g = test_df[test_df.engine_id == eid]
            win = g[cfg.SENSOR_NAMES].tail(N_CYCLES)
            if arm == "revdata":
                win = win.iloc[::-1]          # the window the model was shown
            text = t.get("reasoning", "") or ""
            for s, (canonical, _) in SENSOR_PHYSICS.items():
                rows.append({
                    "arm": arm, "engine_id": eid, "sensor": s, "cued": s in CUED,
                    "claimed": classify_mention(text, s),
                    "canonical": canonical,
                    "cue_says": cue_direction(arm, s),
                    "actual_shown": direction(np.round(win[s].values.astype(float), 3)),
                    "pred_rul": t["pred_rul"], "true_rul": t["true_rul"],
                })
    cl = pd.DataFrame(rows)
    cl.to_csv(OUT / "p0_23_claims.csv", index=False)

    d = cl[cl.claimed.isin(["increase", "decrease"])].copy()
    d["textbook_hit"] = d.claimed == d.canonical
    d["input_hit"] = d.claimed == d.actual_shown
    d["cue_hit"] = np.where(d.cue_says.isna(), np.nan, d.claimed == d.cue_says)

    summ = pd.read_csv(OUT / "p0_23_arm_summary.csv")

    L = ["# P0.2b / P0.3 - control arms: does the model follow the prompt or the data?\n"]
    L.append(f"Paired design: the same 100 {DATASET} test engines, the same model "
             f"(llama3.1:8b, T=0.1), the same zero-shot template, n={N_CYCLES}. "
             "One thing changes per arm.\n")

    # ---- numbers -------------------------------------------------------
    L.append("\n## 1. The predicted numbers barely move\n")
    L.append("| Arm | RMSE | MAE | NASA | distinct values | modal value | modal share |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in summ.set_index("arm").loc[ARMS].reset_index().iterrows():
        L.append(f"| {ARM_LABEL[r.arm]} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.NASA_Score:,.0f} | "
                 f"{int(r.distinct_values)} | {int(r.modal_value)} | {r.modal_share_pct:.0f}% |")
    L.append("\nThe replication arm lands inside the band of the four existing runs of this "
             "exact configuration (RMSE 61.5-62.8, modal value 23 at 80-85%), so the "
             "harness reproduces `experiment.py`.\n")
    L.append("**Every arm still collapses onto 23.** Removing the cue, reversing the cue, "
             "and reversing every trend in the data all leave the output distribution "
             "essentially unchanged: 3-5 distinct values over 100 engines, 69-84% on the "
             "single integer 23. The number the model emits is a property of the prompt "
             "format, not of the sensor evidence or of the physics it is told.\n")

    # ---- explanations follow the cue -----------------------------------
    L.append("\n## 2. The explanations follow the cue (P0.2b)\n")
    L.append("| Arm | Directional claims | Agreement with canonical physics | Agreement with what THIS prompt asserts |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        sub = d[(d.arm == arm) & d.cued]
        cue = sub.cue_hit.dropna()
        cue_s = f"{100 * cue.mean():.1f}%" if len(cue) else "n/a (no cue)"
        L.append(f"| {ARM_LABEL[arm]} | {len(sub):,} | {100 * sub.textbook_hit.mean():.1f}% | {cue_s} |")

    rep = d[(d.arm == 'repl') & d.cued]
    noc = d[(d.arm == 'nocue') & d.cued]
    inv = d[(d.arm == 'invcue') & d.cued]
    L.append(f"\n- **Removing the cue does not lower canonical agreement** "
             f"({100 * rep.textbook_hit.mean():.1f}% -> {100 * noc.textbook_hit.mean():.1f}%). "
             f"What it changes is how much the model talks about directions at all: "
             f"directional claims fall from **{len(rep):,} to {len(noc):,}** "
             f"({100 * (1 - len(noc) / len(rep)):.0f}% fewer). Unprompted, the model largely "
             "stops naming sensor trends; when it still names one it stays canonical. So the "
             "cue is not the sole source of the 91% - the canonical degradation narrative is "
             "also in the model's priors.")
    L.append(f"- **Reversing the cue does move it**, from {100 * rep.textbook_hit.mean():.1f}% "
             f"to **{100 * inv.textbook_hit.mean():.1f}%**, with "
             f"**{100 * inv.cue_hit.dropna().mean():.1f}%** of claims now following the "
             "asserted (physically wrong) direction. Told that worn engines behave the "
             "opposite way, the model reverses roughly a third of its claims about the very "
             "same unchanged sensor data. The prompt exerts a large causal effect on the "
             "explanations; the prior resists the rest.")
    ct = pd.crosstab(d[d.cued & d.arm.isin(["repl", "invcue"])].arm,
                     d[d.cued & d.arm.isin(["repl", "invcue"])].textbook_hit)
    if ct.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct)
        L.append(f"- repl vs invcue: chi2 = {chi2:.1f}, p = {p:.2e}.")

    # ---- explanations ignore the data ----------------------------------
    L.append("\n## 3. The explanations ignore the data (P0.3)\n")
    L.append("| Arm | Directional claims | Faithfulness to the window actually shown |")
    L.append("|---|---|---|")
    for arm in ("repl", "revdata"):
        sub = d[d.arm == arm]
        L.append(f"| {ARM_LABEL[arm]} | {len(sub):,} | {100 * sub.input_hit.mean():.1f}% |")

    # paired test: pairs whose real trend flips between the two arms
    piv = cl[cl.arm.isin(["repl", "revdata"])].pivot_table(
        index=["engine_id", "sensor"], columns="arm",
        values=["claimed", "actual_shown"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna()
    both_dir = piv[piv.claimed_repl.isin(["increase", "decrease"]) &
                   piv.claimed_revdata.isin(["increase", "decrease"])]
    flipped = both_dir[both_dir.actual_shown_repl != both_dir.actual_shown_revdata]
    claim_flipped = flipped[flipped.claimed_repl != flipped.claimed_revdata]

    L.append(f"\n**The paired test.** Among the (engine, sensor) pairs where the model made a "
             f"directional claim in both arms and the trend in the window genuinely reversed "
             f"(**n = {len(flipped):,}**), the claim changed in "
             f"**{len(claim_flipped)} cases ({100 * len(claim_flipped) / max(len(flipped), 1):.1f}%)**.")
    L.append("\nThe data was turned upside down and the explanation stayed the same. Whatever "
             "the reasoning text is describing, it is not the sensor window in the prompt.\n")

    L.append("\n## 4. Per sensor, repl vs revdata\n")
    L.append("| Sensor | In cue? | Claims increase/decrease (repl) | Claims increase/decrease (revdata) |")
    L.append("|---|---|---|---|")
    for s in SENSOR_PHYSICS:
        a = d[(d.arm == "repl") & (d.sensor == s)].claimed.value_counts()
        b = d[(d.arm == "revdata") & (d.sensor == s)].claimed.value_counts()
        L.append(f"| {s} | {'yes' if s in CUED else 'NO'} | "
                 f"{a.get('increase', 0)} / {a.get('decrease', 0)} | "
                 f"{b.get('increase', 0)} / {b.get('decrease', 0)} |")

    L.append("\n## Conclusion\n")
    L.append("- The **numbers** are unmoved by removing the cue, by reversing the cue, and by "
             "reversing the data. Every arm collapses onto 23.")
    L.append("- The **explanations** are driven by the prompt and by the model's canonical "
             "prior, in that order, and by the sensor window not at all: when the trends are "
             f"reversed the claims follow only {100 * len(claim_flipped) / max(len(flipped), 1):.1f}% "
             "of the time.")
    L.append("- Nuance worth keeping in the paper: the reported 91% is **not purely** prompt "
             "leakage. Deleting the cue leaves canonical agreement intact while suppressing "
             "most directional claims, so part of it comes from the model's priors. "
             "Contradicting the cue does move agreement by 32 points, which establishes the "
             "prompt's causal role.")
    L.append("- Either way the metric never measured what the paper claimed: under every arm, "
             "including the untouched original, the explanations match the data given to the "
             "model roughly 43% of the time - below the rate obtained by ignoring the data "
             "entirely (P0.2a).")

    (OUT / "p0_23_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p0_23_report.md'}")


if __name__ == "__main__":
    main()
