"""
P1 analysis - consolidate the model sweep against the reference and the baselines.

Puts every model run at the canonical configuration (FD001, zero-shot, n=30,
100 engines, T=0.1) side by side with:
  * llama3.1:8b Q4_K_M, the paper's model, pooled over its 5 existing runs of
    this exact configuration (4 in results/, plus the P0 `repl` arm);
  * the no-skill constant and the repaired supervised baselines.

Outputs (results_p1/):
  p1_comparison.csv
  p1_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import experiment as ex  # noqa: E402

OUT = EXP / "results_p1"
P0 = EXP / "results_p0"
RUL_CAP = 125

REF_FILES = [EXP / "results" / f"traces_FD001_zero_shot_k0_n30{s}.json"
             for s in ("", "_run2", "_run3", "_run4")] + \
            [P0 / "traces_p0_FD001_zs_n30_repl.json"]


def describe(y, p, label, extra=None):
    vals, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    # Ranking a predictor whose output range is floating-point noise is
    # meaningless; report rho as n/a instead (see P0.1b).
    degenerate = bool(p.max() - p.min() < 1e-3)
    rho = spearmanr(p, y) if (len(vals) > 1 and not degenerate) else None
    d = {"model": label, "n": len(p),
         "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
         "NASA_Score": ex.nasa_score(y, p),
         "distinct_values": int(len(vals)),
         "modal_value": float(vals[counts.argmax()]),
         "modal_share_pct": 100 * counts.max() / len(p),
         "entropy_bits": float(-(q * np.log2(q)).sum()),
         "pred_sd": float(p.std(ddof=1)),
         "spearman_rho": float(rho.statistic) if rho else np.nan,
         "spearman_p": float(rho.pvalue) if rho else np.nan}
    if extra:
        d.update(extra)
    return d


def main():
    rows = []

    # -- the paper's model, pooled over its five runs ----------------------
    ref = []
    for f in REF_FILES:
        if not f.exists():
            continue
        tr = json.loads(f.read_text(encoding="utf-8"))
        y = np.array([t["true_rul"] for t in tr], float)
        p = np.array([t["pred_rul"] for t in tr], float)
        ref.append(describe(y, p, "llama3.1:8b Q4_K_M (reference)"))
    if ref:
        rd = pd.DataFrame(ref)
        rows.append({"model": f"llama3.1:8b Q4_K_M [{len(ref)} runs, mean]",
                     "n": int(rd.n.mean()),
                     **{c: float(rd[c].mean()) for c in
                        ("RMSE", "MAE", "NASA_Score", "distinct_values",
                         "modal_value", "modal_share_pct", "entropy_bits",
                         "pred_sd", "spearman_rho", "spearman_p")},
                     "spread": f"RMSE {rd.RMSE.min():.1f}-{rd.RMSE.max():.1f}"})

    # -- the swept models --------------------------------------------------
    for f in sorted(OUT.glob("traces_p1_FD001_zs_n30_*.json")):
        tr = json.loads(f.read_text(encoding="utf-8"))
        model = tr[0].get("model", f.stem)
        usable = [t for t in tr if t.get("pred_rul") is not None]
        if not usable:
            continue
        y = np.array([t["true_rul"] for t in usable], float)
        p = np.array([t["pred_rul"] for t in usable], float)
        rows.append(describe(y, p, model, {
            "spread": f"parse_ok {100 * np.mean([t.get('parse_ok', True) for t in tr]):.0f}%"}))

    # -- baselines ---------------------------------------------------------
    bp = OUT / "p2_1_baseline_predictions.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        b = b[b.dataset == "FD001"]
        y = b.true_rul.values.astype(float)
        for col, lab in (("const_trainmean", "constant = train mean (no skill)"),
                         ("RandomForest", "RandomForest"),
                         ("XGBoost", "XGBoost"),
                         ("LSTM_original", "LSTM (as shipped)"),
                         ("LSTM_fixed", "LSTM (repaired)")):
            if col in b:
                rows.append(describe(y, b[col].values.astype(float), lab,
                                     {"spread": "baseline"}))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p1_comparison.csv", index=False)

    L = ["# P1 - Is the collapse specific to llama3.1:8b Q4_K_M?\n"]
    L.append("Canonical configuration for every LLM row: FD001, zero-shot, n=30, "
             "100 test engines, T=0.1, shipped prompt template. Only the model changes.\n")
    L.append("\n| Model | engines | RMSE | MAE | NASA | distinct | modal (share) | "
             "entropy (bits) | rho |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        rho = "n/a" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.3f}"
        partial = "" if r.n >= 100 else " **PARTIAL**"
        L.append(f"| {r.model}{partial} | {r.n:.0f} | {r.RMSE:.2f} | {r.MAE:.2f} | "
                 f"{r.NASA_Score:,.0f} | {r.distinct_values:.0f} | "
                 f"{r.modal_value:.0f} ({r.modal_share_pct:.0f}%) | "
                 f"{r.entropy_bits:.2f} | {rho} |")
    incomplete = df[df.n < 100]
    if len(incomplete):
        L.append(f"\n> **Warning:** {len(incomplete)} run(s) still in progress, computed on "
                 f"fewer than 100 engines "
                 f"({', '.join(f'{r.model}: n={r.n:.0f}' for _, r in incomplete.iterrows())}). "
                 "Rerun this script once the sweep finishes.")

    llm = df[~df.model.isin(["constant = train mean (no skill)", "RandomForest", "XGBoost",
                             "LSTM (as shipped)", "LSTM (repaired)"])]
    const = df[df.model == "constant = train mean (no skill)"]
    L.append("\n## Reading\n")
    if len(llm):
        L.append(f"- Distinct values over 100 engines, across every model tested: "
                 f"**{llm.distinct_values.min():.0f}-{llm.distinct_values.max():.0f}** "
                 f"(a genuine regressor gives up to 100). Output entropy "
                 f"{llm.entropy_bits.min():.2f}-{llm.entropy_bits.max():.2f} bits against a "
                 f"ceiling of {np.log2(100):.2f}.")
        if len(const):
            c = float(const.RMSE.iloc[0])
            beat = llm[llm.RMSE < c]
            L.append(f"- Models beating the no-skill constant (RMSE {c:.2f}) on RMSE: "
                     f"**{len(beat)}/{len(llm)}**"
                     + (f" ({', '.join(beat.model)})" if len(beat) else ""))
    L.append("- If the collapse were an artefact of one model or one quantisation, swapping "
             "the model would relieve it.")

    (OUT / "p1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p1_report.md'}")


if __name__ == "__main__":
    main()
