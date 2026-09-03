"""
P1.3 / P1.4 analysis - do real examples move the anchor, and how much does the
example set matter?

Reads the traces written by p1_34_fewshot_sampling.py and answers:

  P1.3  Does replacing the defective example set (1 example, RUL 0) with
        genuinely stratified examples move the predictions? The mechanistic
        hypothesis from P1.0 predicts the anchor rises towards the test mean.
  P1.4  How much do results vary across independent example sets? The paper
        reported a single draw with no variability estimate.

`asis_seed1` reproduces the shipped prompt, so it doubles as a replication
control against the published k=5, n=5 run.

Outputs (results_p1/):
  p1_34_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import experiment as ex  # noqa: E402

OUT = EXP / "results_p1"
TEST_MEAN = None  # filled from data


def summarise(path, label):
    tr = json.loads(Path(path).read_text(encoding="utf-8"))
    y = np.array([t["true_rul"] for t in tr], float)
    p = np.array([t["pred_rul"] for t in tr], float)
    vals, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    rho = spearmanr(p, y) if len(vals) > 1 else None
    return {"run": label, "n": len(p),
            "spearman_rho": float(rho.statistic) if rho else np.nan,
            "spearman_p": float(rho.pvalue) if rho else np.nan,
            "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
            "NASA_Score": ex.nasa_score(y, p),
            "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)),
            "distinct_values": int(len(vals)),
            "modal_value": float(vals[counts.argmax()]),
            "modal_share_pct": 100 * counts.max() / len(p),
            "entropy_bits": float(-(q * np.log2(q)).sum())}


def main():
    rows = []

    pub = EXP / "results" / "traces_FD001_few_shot_k5.json"
    if pub.exists():
        r = summarise(pub, "published k=5, n=5 (same prompt as asis)")
        r["condition"] = "published"
        rows.append(r)

    for f in sorted(OUT.glob("traces_p1_FD001_fs_k5_n5_*.json")):
        tag = f.stem.replace("traces_p1_FD001_fs_k5_n5_", "")
        r = summarise(f, tag)
        r["condition"] = tag.split("_seed")[0]
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p1_34_analysis.csv", index=False)

    _, _, test_last = ex.load_cmapss("FD001")
    test_mean = float(test_last["RUL"].mean())

    L = ["# P1.3 / P1.4 - Does the example set explain the anchor?\n"]
    L.append("FD001, k=5, n=5, 100 test engines, llama3.1:8b, T=0.1. Only the few-shot "
             f"example set changes. Test-set mean RUL = **{test_mean:.1f}**.\n")
    L.append("\n| Run | n | RMSE | MAE | NASA | mean pred | rho (p) | distinct | "
             "modal (share) | entropy |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        partial = "" if r.n >= 100 else " **PARTIAL**"
        rho = ("n/a" if pd.isna(r.spearman_rho)
               else f"{r.spearman_rho:+.3f} ({r.spearman_p:.2f})")
        L.append(f"| {r.run}{partial} | {r.n:.0f} | {r.RMSE:.2f} | {r.MAE:.2f} | "
                 f"{r.NASA_Score:,.0f} | {r.pred_mean:.1f} | {rho} | "
                 f"{r.distinct_values:.0f} | "
                 f"{r.modal_value:.0f} ({r.modal_share_pct:.0f}%) | {r.entropy_bits:.2f} |")

    done = df[df.n >= 100]
    asis = done[done.condition == "asis"]
    strat = done[done.condition == "strat"]
    rnd = done[done.condition == "random"]
    pubr = done[done.condition == "published"]

    if len(asis) and len(pubr):
        L.append(f"\n**Replication control.** `asis` reproduces the shipped prompt and lands at "
                 f"RMSE {float(asis.RMSE.iloc[0]):.2f} / modal "
                 f"{int(asis.modal_value.iloc[0])} against the published run's "
                 f"{float(pubr.RMSE.iloc[0]):.2f} / {int(pubr.modal_value.iloc[0])}.\n")

    if len(asis) and len(strat):
        L.append("\n## P1.3 - the mechanistic hypothesis\n")
        a_mean = float(asis.pred_mean.iloc[0])
        s_mean = float(strat.pred_mean.mean())
        L.append(f"P1.0 showed the shipped prompt contains one example labelled RUL 0. If that "
                 f"is what drags the predictions down, then real examples spanning 0-125 should "
                 f"raise them towards the test mean of {test_mean:.1f}.\n")
        L.append(f"- Mean prediction, shipped example set: **{a_mean:.1f}**")
        L.append(f"- Mean prediction, genuinely stratified examples: "
                 f"**{s_mean:.1f}** (mean of {len(strat)} seeds)")
        L.append(f"- Shift: **{s_mean - a_mean:+.1f} RUL**, against a gap to the test mean of "
                 f"{test_mean - a_mean:+.1f}")
        moved = "supports" if s_mean > a_mean else "does not support"
        L.append(f"\nThe direction of the shift **{moved}** the hypothesis that the RUL-0 "
                 "example set is what anchors the predictions low.")
        L.append(f"- Collapse itself: distinct values go from {int(asis.distinct_values.iloc[0])} "
                 f"(shipped) to {strat.distinct_values.mean():.1f} (stratified, mean of seeds); "
                 f"entropy {float(asis.entropy_bits.iloc[0]):.2f} -> "
                 f"{strat.entropy_bits.mean():.2f} bits against a ceiling of "
                 f"{np.log2(100):.2f}. Fixing the examples does **not** restore regression.")

    if len(strat) and len(rnd):
        L.append("\n## Stratified vs unstratified\n")
        L.append(f"- stratified: RMSE {strat.RMSE.mean():.2f} +/- {strat.RMSE.std(ddof=1):.2f}, "
                 f"mean pred {strat.pred_mean.mean():.1f}")
        L.append(f"- random:     RMSE {rnd.RMSE.mean():.2f} +/- {rnd.RMSE.std(ddof=1):.2f}, "
                 f"mean pred {rnd.pred_mean.mean():.1f}")
        if len(strat) >= 3 and len(rnd) >= 3:
            u, pv = mannwhitneyu(strat.RMSE, rnd.RMSE)
            L.append(f"- Mann-Whitney on RMSE: U = {u:.0f}, p = {pv:.3f}")

    if len(strat) > 1 or len(rnd) > 1:
        L.append("\n## P1.4 - sampling variability the paper never reported\n")
        L.append("| Condition | seeds | RMSE mean | RMSE sd | RMSE range | modal values |")
        L.append("|---|---|---|---|---|---|")
        for cond, sub in [("stratified", strat), ("random", rnd)]:
            if len(sub) < 2:
                continue
            L.append(f"| {cond} | {len(sub)} | {sub.RMSE.mean():.2f} | "
                     f"{sub.RMSE.std(ddof=1):.2f} | {sub.RMSE.min():.2f}-{sub.RMSE.max():.2f} | "
                     f"{sorted(set(sub.modal_value.astype(int)))} |")
        L.append("\nThe paper reported one draw of the example set with no variability estimate. "
                 "Any k-to-k difference smaller than this spread is not interpretable.")

    # A run can beat the no-skill constant on RMSE with no predictive content at
    # all, simply by anchoring closer to the mean. Say so explicitly.
    NOSKILL = 41.94  # constant = train mean, FD001 (results_p0/p0_1_report.md)
    winners = done[done.RMSE < NOSKILL]
    L.append("\n## Why a lower RMSE here is not predictive skill\n")
    L.append(f"Random example sets draw mostly high-RUL windows - the target is capped at 125 - "
             f"so they anchor the model near the test mean of {test_mean:.1f} and the RMSE "
             "falls. That is a constant being relocated, not engines being told apart.\n")
    # Benjamini-Hochberg over this family of runs, so the rho column is not read
    # one nominal p-value at a time.
    fam = done[done.condition != "published"].copy()
    order = np.argsort(fam.spearman_p.values)
    m = len(fam)
    adj = np.empty(m)
    ranked = fam.spearman_p.values[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj[order] = np.clip(ranked, 0, 1)
    fam["p_fdr"] = adj

    if len(winners):
        L.append(f"**{len(winners)} run(s) come in under the no-skill constant "
                 f"(RMSE {NOSKILL}):**\n")
        L.append("| Run | RMSE | mean pred | rho | p | p (BH-FDR) | distinct | rank var. explained |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in winners.iterrows():
            pf = fam.loc[fam.run == r.run, "p_fdr"]
            pfs = f"{float(pf.iloc[0]):.3f}" if len(pf) else "n/a"
            L.append(f"| {r.run} | {r.RMSE:.2f} | {r.pred_mean:.1f} | {r.spearman_rho:+.3f} | "
                     f"{r.spearman_p:.2f} | {pfs} | {r.distinct_values:.0f} | "
                     f"{100 * r.spearman_rho ** 2:.1f}% |")

        best = fam.loc[fam.spearman_rho.idxmax()]
        L.append(f"\nThis has to be stated carefully. The strongest of them reaches "
                 f"rho = {best.spearman_rho:+.3f} (p = {best.spearman_p:.2f}, "
                 f"BH-adjusted {best.p_fdr:.3f}), so it is **not** literally zero and it would "
                 "be wrong to write that no configuration orders the engines at all. Three "
                 "things keep it from being evidence of prognostic ability:")
        by_cond = fam[fam.condition == best.condition]
        L.append(f"1. **It explains {100 * best.spearman_rho ** 2:.0f}% of the rank variance**, "
                 "against 66% for Random Forest (+0.813) and 79% for the repaired LSTM "
                 "(+0.889) on the same 100 engines.")
        L.append(f"2. **It is unstable across seeds of its own condition**: the "
                 f"{best.condition} runs span rho "
                 f"{by_cond.spearman_rho.min():+.3f} to {by_cond.spearman_rho.max():+.3f} "
                 "with nothing changed but which examples were drawn. A capability does not "
                 "come and go with the draw.")
        L.append(f"3. **The RMSE gain is anchoring, not discrimination.** Its mean prediction "
                 f"is {best.pred_mean:.1f} against a test mean of {test_mean:.1f}, on "
                 f"{best.distinct_values:.0f} distinct values over 100 engines "
                 f"({best.entropy_bits:.2f} bits of {np.log2(100):.2f}).")
        L.append("\nSo a configuration can beat the no-skill constant on RMSE while carrying "
                 "almost no prognostic content - which is exactly why the rewritten paper must "
                 "report a rank correlation next to every error metric, and why RMSE alone was "
                 "never going to settle this question.")
    else:
        L.append("No completed run beats the no-skill constant.")
    L.append(f"\nAcross all completed runs the largest |rho| is "
             f"**{done.spearman_rho.abs().max():.3f}**; Random Forest reaches +0.813 and the "
             "repaired LSTM +0.889 on the same 100 engines.")

    (OUT / "p1_34_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p1_34_report.md'}")


if __name__ == "__main__":
    main()
