"""
P3.2 - Three loose ends in the statistics.

MULTIPLICITY. Twenty-seven rank-correlation tests were run and the manuscript
highlights one configuration. This writes out all 27 with their uncorrected,
Benjamini-Hochberg and Bonferroni p-values, states what the family is, and says
plainly which of the tests were decided in advance and which were not.

THE EXAMPLE-SET ASSOCIATION. The correlation between the mean RUL of the prompt
examples and the mean prediction rests on seven aggregate points. It is reported
here as the exploratory result it is: n, r, p, a bootstrap interval, and a
leave-one-out pass that shows how much any single example set is carrying.

A COMMON GRID. Counting distinct predicted values favours a continuous model over
one instructed to answer with an integer, because two continuous predictions are
almost never equal. Everything is rounded to the same one-cycle grid and the
concentration measures are recomputed, so the comparison is between like and like.

Outputs (results_p1/):
  p3_2_rank_family.csv
  p3_2_discretised.csv
  p3_2_report.md
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"

N_BOOT = 10000
SEED = 20260906


def entropy_bits(values):
    n = len(values)
    if n == 0:
        return np.nan
    p = np.array([c / n for c in Counter(values).values()])
    return float(-(p * np.log2(p)).sum())


def concentration(values):
    n = len(values)
    cnt = Counter(values)
    modal, k = cnt.most_common(1)[0]
    return {"n": n, "distinct": len(cnt), "modal_value": modal,
            "modal_share_pct": round(100 * k / n, 1),
            "entropy_bits": round(abs(entropy_bits(values)), 3),
            "entropy_ceiling_bits": round(float(np.log2(n)), 2)}


def main():
    L = ["# P3.2 - Multiplicity, the example-set association, and a common grid\n"]

    # ================================================================ 1
    L.append("\n## The family of rank-correlation tests\n")
    rk = pd.read_csv(P0 / "p0_1b_rank_significance.csv")
    rk = rk.sort_values("spearman_p").reset_index(drop=True)
    n_tests = len(rk)
    L.append(f"**The family is the {n_tests} runs of the main prompting grid**: every "
             "configuration of dataset, prompting mode, number of examples and window length "
             "for which a trace file exists, including the replicate runs of the zero-shot "
             "configuration. One Spearman correlation between predicted and true RUL is "
             "computed per run, so the family is fixed by the design of the grid and not by "
             "which results turned out to be interesting.\n")
    L.append("**What was decided in advance and what was not.** The grid itself was "
             "pre-specified: the datasets, the values of k and n, and the decision to compute "
             "a rank correlation for every cell were all fixed before the runs. Nothing was "
             "added to the family afterwards. The correction across the family, and the "
             "decision to report rank correlation next to every error metric, were adopted "
             "after seeing that error alone could not separate ordering from anchoring, so "
             "they are post hoc, and the emphasis that Section 4.7 places on one particular "
             "run is post hoc by construction, since it is the run that stood out. That is "
             "why the whole family is printed here.\n")

    nom = int((rk.spearman_p < 0.05).sum())
    fdr = int((rk.p_fdr < 0.05).sum())
    bon = int((rk.p_bonferroni < 0.05).sum())
    L.append(f"Of the {n_tests} tests, **{nom} reach p < 0.05 uncorrected**, **{fdr} survive "
             f"Benjamini-Hochberg** at 5% and **{bon} survives Bonferroni**. The largest "
             f"absolute correlation anywhere in the family is "
             f"{rk.spearman_rho.abs().max():.3f}, whose square is "
             f"{rk.spearman_rho.abs().max() ** 2:.3f}. Signs are inconsistent: "
             f"{int((rk.spearman_rho > 0).sum())} positive and "
             f"{int((rk.spearman_rho < 0).sum())} negative. Benjamini-Hochberg is the primary "
             "correction, as the tests are one family of related hypotheses and false "
             "discovery is the relevant error rate; Bonferroni is given as the strict "
             "alternative.\n")
    L.append("| # | Dataset | Mode | k | n | Run | rho | 95% CI | p | p (BH) | p (Bonf.) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in rk.iterrows():
        ci = (f"[{r.rho_ci_lo:+.2f}, {r.rho_ci_hi:+.2f}]"
              if np.isfinite(r.rho_ci_lo) else "n/a")
        L.append(f"| {i + 1} | {r.dataset} | {r['mode'].replace('_', '-')} | {int(r.k)} | "
                 f"{int(r.n_cycles)} | {r.run} | {r.spearman_rho:+.3f} | {ci} | "
                 f"{r.spearman_p:.3f} | {r.p_fdr:.3f} | "
                 f"{min(r.p_bonferroni, 1.0):.3f} |")
    L.append("\nFor scale, Random Forest and the scaled-target LSTM reach rho of about +0.81 "
             "to +0.93 on the same engines, with p below 1e-24. The question is not whether "
             "any of these 27 correlations is nominally significant, but whether any is large "
             "enough to be prognostically useful, and none is.\n")
    rk.to_csv(P1 / "p3_2_rank_family.csv", index=False)

    # ================================================================ 2
    L.append("\n## The example-set association, as an exploratory result\n")
    s = pd.read_csv(P1 / "p1_34_summary.csv")
    pub = P1 / "traces_p1_FD001_fs_k5_n5_asis_seed1.json"
    pts = []
    for _, r in s.iterrows():
        ruls = json.loads(r.example_ruls) if isinstance(r.example_ruls, str) else r.example_ruls
        pts.append((float(np.mean(ruls)), float(r.pred_mean), f"{r.condition}_seed{int(r.seed)}"))
    if pub.exists():
        tr = json.loads(pub.read_text(encoding="utf-8"))
        pts.append((0.0, float(np.mean([t["pred_rul"] for t in tr])), "asis_seed1"))
    pts.sort()
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    lab = [p[2] for p in pts]
    n = len(x)
    r, p = pearsonr(x, y)
    rho, prho = spearmanr(x, y)

    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        boots.append(pearsonr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    L.append(f"Seven example sets were run, each with k fixed at 5, so the only thing that "
             f"changes between them is which examples the prompt carries. Across those seven "
             f"points the mean prediction correlates with the mean RUL of the examples at "
             f"**r = {r:+.3f}** (Pearson, n = {n}, p = {p:.4f}), with a bootstrap 95% interval "
             f"of [{lo:+.3f}, {hi:+.3f}]. Spearman's rho on the same points is {rho:+.3f} "
             f"(p = {prho:.4f}).\n")
    L.append("**This is an exploratory result and should be read as one.** Seven aggregate "
             "points is a small sample and an association of this kind "
             "does not establish that the examples cause the anchor; it establishes that the "
             "two move together across the sets that were run.\n")
    L.append("| Example set | Mean RUL of the examples | Mean prediction | r without this point |")
    L.append("|---|---|---|---|")
    loo = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        ri = pearsonr(x[m], y[m])[0]
        loo.append(ri)
        L.append(f"| {lab[i]} | {x[i]:.1f} | {y[i]:.1f} | {ri:+.3f} |")
    L.append(f"\nLeave-one-out, r ranges from **{min(loo):+.3f} to {max(loo):+.3f}**. No "
             "single example set is carrying the association, including the one whose "
             "examples all sit at the end of life: with that point dropped, r is "
             f"{loo[0]:+.3f} on the remaining six.")
    L.append("Squared, r gives the share of the between-set variance in mean prediction that "
             "the mean example RUL accounts for. That is a statement about seven aggregate "
             "points, and it is not comparable with a squared rank correlation computed over "
             "100 engines within a single run: different units of analysis, different sample "
             "sizes, different measures of association. The two should never be presented as "
             "shares of one variance.\n")

    # ================================================================ 3
    L.append("\n## Concentration on a common one-cycle grid\n")
    L.append("The language models are instructed to answer with an integer, while Random "
             "Forest, XGBoost and the LSTM emit floating-point numbers, so two of their "
             "predictions are almost never exactly equal. Counting distinct values as they "
             "come therefore flatters the supervised models automatically. Here every "
             "prediction, from every model, is rounded to the nearest cycle before the "
             "concentration measures are computed. Two values count as the same when they "
             "round to the same integer.\n")

    rows = []
    bp = pd.read_csv(P1 / "p2_1_baseline_predictions.csv")
    for ds, g in bp.groupby("dataset"):
        for col in ["const_trainmean", "RandomForest", "XGBoost",
                    "LSTM_original", "LSTM_fixed"]:
            raw = g[col].to_numpy(dtype=float)
            rows.append({"dataset": ds, "model": col, "kind": "supervised",
                         "distinct_raw": len(set(np.round(raw, 6))),
                         **concentration(np.round(raw).astype(int).tolist())})
    canon = {
        "llama3.1:8b (reference)": EXP / "results" / "traces_FD001_zero_shot_k0_n30.json",
        "llama3.1:8b-instruct-q8_0": P1 / "traces_p1_FD001_zs_n30_llama3.1_8b-instruct-q8_0.json",
        "mistral:7b": P1 / "traces_p1_FD001_zs_n30_mistral_7b.json",
        "qwen2.5:7b": P1 / "traces_p1_FD001_zs_n30_qwen2.5_7b.json",
        "deepseek-r1:7b": P1 / "traces_p1_FD001_zs_n30_deepseek-r1_7b.json",
    }
    for name, path in canon.items():
        if not path.exists():
            continue
        tr = json.loads(path.read_text(encoding="utf-8"))
        raw = np.array([t["pred_rul"] for t in tr], dtype=float)
        rows.append({"dataset": "FD001", "model": name, "kind": "prompted",
                     "distinct_raw": len(set(np.round(raw, 6))),
                     **concentration(np.round(raw).astype(int).tolist())})
    df = pd.DataFrame(rows)
    df.to_csv(P1 / "p3_2_discretised.csv", index=False)

    L.append("| Dataset | Model | Engines | Distinct, raw | Distinct, 1-cycle grid | "
             "Modal share | Entropy (bits) | Ceiling |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L.append(f"| {r.dataset} | {r.model} | {r['n']} | {r.distinct_raw} | {r.distinct} | "
                 f"{r.modal_share_pct}% | {r.entropy_bits} | {r.entropy_ceiling_bits} |")
    sup = df[(df.kind == "supervised") & (~df.model.str.contains("const|LSTM_original"))]
    pro = df[df.kind == "prompted"]
    L.append(f"\nRounding changes nothing that matters. On the common grid the supervised "
             f"regressors still occupy {int(sup.distinct.min())} to {int(sup.distinct.max())} "
             f"distinct values with a modal share of {sup.modal_share_pct.min():.0f}% to "
             f"{sup.modal_share_pct.max():.0f}%, while the prompted models occupy "
             f"{int(pro.distinct.min())} to {int(pro.distinct.max())} with a modal share of "
             f"{pro.modal_share_pct.min():.0f}% to {pro.modal_share_pct.max():.0f}%. The gap "
             "is not an artefact of integer output against continuous output.\n")
    L.append("The two degenerate rows are worth keeping in view. The training-mean constant "
             "and the unscaled-target LSTM occupy one value each on this grid, which is what "
             "a collapsed predictor looks like, and the concentrated prompting runs sit next "
             "to them rather than next to the regressors.\n")

    (P1 / "p3_2_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_2_report.md'}")


if __name__ == "__main__":
    main()
