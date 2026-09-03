"""
P0.1b - Does ANY configuration order the engines?

P0.1 showed that a minority of configurations reach a nominally significant
Spearman rho. This script checks whether that survives scrutiny:

  1. Benjamini-Hochberg FDR correction across the whole family of tests.
  2. Bootstrap 95% CI on rho (10,000 resamples).
  3. Sign consistency: a genuine ordinal signal cannot be positive in one
     configuration and negative in another.
  4. Reference point: the same rho computed for Random Forest and LSTM,
     which DO regress. Without this the LLM's rho has no scale.

Outputs (results_p0/):
  p0_1b_rank_significance.csv
  p0_1b_baseline_predictions.csv
  p0_1b_report.md
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

OUT = EXP / "results_p0"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260901)
N_BOOT = 10000


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def boot_rho_ci(p, y, n_boot=N_BOOT):
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        pp, yy = p[idx], y[idx]
        if len(np.unique(pp)) < 2 or len(np.unique(yy)) < 2:
            vals[i] = 0.0
        else:
            vals[i] = spearmanr(pp, yy).statistic
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    cfg = pd.read_csv(OUT / "p0_1_llm_configs.csv")

    # -- bootstrap CI per configuration ------------------------------------
    los, his = [], []
    for _, r in cfg.iterrows():
        tr = json.loads((EXP / "results" / r.file).read_text(encoding="utf-8"))
        y = np.array([t["true_rul"] for t in tr], float)
        p = np.array([t["pred_rul"] for t in tr], float)
        lo, hi = boot_rho_ci(p, y)
        los.append(lo)
        his.append(hi)
    cfg["rho_ci_lo"] = los
    cfg["rho_ci_hi"] = his
    cfg["p_fdr"] = bh_fdr(cfg.spearman_p.values)
    cfg["sig_fdr_5pct"] = cfg.p_fdr < 0.05
    cfg["p_bonferroni"] = np.clip(cfg.spearman_p * len(cfg), 0, 1)
    cfg["sig_bonf_5pct"] = cfg.p_bonferroni < 0.05

    # -- baselines that actually regress, as a scale reference -------------
    import experiment as ex
    import config as c

    base_rows, base_preds = [], []
    for ds in ("FD001", "FD003"):
        train_df, test_df, test_last = ex.load_cmapss(ds)
        fitted = {"RandomForest": ex.run_rf(train_df, test_df, test_last),
                  "LSTM": ex.run_lstm(train_df, test_df, test_last)}
        for name, (pp, yy) in fitted.items():
            if pp is None:
                continue
            pp = np.asarray(pp, float)
            yy = np.asarray(yy, float)
            spread = float(pp.max() - pp.min())
            # A predictor whose whole output range is floating-point noise is
            # constant. Ranking that noise produces a rho with no meaning, so
            # it is reported as n/a rather than as a correlation.
            degenerate = spread < 1e-3
            row = {"dataset": ds, "model": name,
                   "RMSE": ex.rmse(yy, pp), "MAE": ex.mae(yy, pp),
                   "NASA_Score": ex.nasa_score(yy, pp),
                   "pred_min": float(pp.min()), "pred_max": float(pp.max()),
                   "pred_sd": float(pp.std(ddof=1)), "pred_range": spread,
                   "degenerate_constant": degenerate,
                   "distinct_values": int(len(np.unique(np.round(pp, 3))))}
            if degenerate:
                row.update({"spearman_rho": np.nan, "spearman_p": np.nan,
                            "rho_ci_lo": np.nan, "rho_ci_hi": np.nan})
            else:
                r = spearmanr(pp, yy)
                lo, hi = boot_rho_ci(pp, yy)
                row.update({"spearman_rho": float(r.statistic),
                            "spearman_p": float(r.pvalue),
                            "rho_ci_lo": lo, "rho_ci_hi": hi})
            base_rows.append(row)

        rf_p, rf_y = fitted["RandomForest"]
        ls_p, _ = fitted["LSTM"]

        for eid, t, rp, lp in zip(test_last.engine_id.values, rf_y, rf_p,
                                  ls_p if ls_p is not None else [np.nan] * len(rf_p)):
            base_preds.append({"dataset": ds, "engine_id": int(eid), "true_rul": float(t),
                               "rf_pred": float(rp), "lstm_pred": float(lp)})

    base_df = pd.DataFrame(base_rows)
    pd.DataFrame(base_preds).to_csv(OUT / "p0_1b_baseline_predictions.csv", index=False)
    cfg.to_csv(OUT / "p0_1b_rank_significance.csv", index=False)

    # -- report ------------------------------------------------------------
    L = ["# P0.1b - Rank correlation: significance, effect size, scale reference\n"]
    L.append(f"Family of {len(cfg)} tests (all trace files, both datasets). "
             "Bootstrap: 10,000 resamples.\n")

    L.append("\n## Reference: the supervised baselines\n")
    L.append("| Dataset | Model | RMSE | NASA | rho | 95% CI | p | pred range | distinct preds |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in base_df.iterrows():
        if r.degenerate_constant:
            rho_s, ci_s, p_s = "n/a", "n/a", "n/a"
        else:
            rho_s = f"{r.spearman_rho:+.3f}"
            ci_s = f"[{r.rho_ci_lo:+.2f}, {r.rho_ci_hi:+.2f}]"
            p_s = f"{r.spearman_p:.2e}"
        L.append(f"| {r.dataset} | {r.model} | {r.RMSE:.2f} | {r.NASA_Score:,.0f} | "
                 f"{rho_s} | {ci_s} | {p_s} | "
                 f"{r.pred_min:.3f}-{r.pred_max:.3f} | {r.distinct_values} |")

    deg = base_df[base_df.degenerate_constant]
    if len(deg):
        L.append("\n> **The LSTM baseline is itself a constant predictor.** Its entire "
                 "output range across the 100 test engines is below 1e-3, i.e. floating-point "
                 "noise: it emits one number for every engine "
                 + "; ".join(f"({r.dataset}: {r.pred_min:.4f})" for _, r in deg.iterrows())
                 + ". Spearman rho is therefore undefined for it and is reported as n/a - "
                 "ranking that noise yields a large negative rho that means nothing. "
                 "This invalidates the submitted paper's framing of the LLM as "
                 "'competitive with the LSTM baseline': the comparison was between two "
                 "constant predictors, and the LLM's constant was the worse one.\n")

    L.append("\n## LLM configurations, corrected for multiple testing\n")
    L.append("| Dataset | Config | rho | 95% CI | p raw | p FDR | p Bonf | sig (FDR) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in cfg.sort_values(["dataset", "spearman_p"]).iterrows():
        tag = (f"{'ZS' if r['mode'] == 'zero_shot' else 'FS k=' + str(r.k)}, n={r.n_cycles}"
               + ("" if r.run == "run1" else f" [{r.run}]"))
        L.append(f"| {r.dataset} | {tag} | {r.spearman_rho:+.3f} | "
                 f"[{r.rho_ci_lo:+.2f}, {r.rho_ci_hi:+.2f}] | {r.spearman_p:.3f} | "
                 f"{r.p_fdr:.3f} | {r.p_bonferroni:.3f} | {'YES' if r.sig_fdr_5pct else 'no'} |")

    n_raw = int((cfg.spearman_p < 0.05).sum())
    n_fdr = int(cfg.sig_fdr_5pct.sum())
    n_bonf = int(cfg.sig_bonf_5pct.sum())
    pos = cfg[cfg.sig_fdr_5pct & (cfg.spearman_rho > 0)]
    neg = cfg[cfg.sig_fdr_5pct & (cfg.spearman_rho < 0)]

    L.append("\n## Verdict\n")
    L.append(f"- Nominally significant (p<0.05, uncorrected): **{n_raw}/{len(cfg)}**. "
             f"Expected by chance alone at this family size: {0.05 * len(cfg):.1f}.")
    L.append(f"- Surviving Benjamini-Hochberg FDR at 5%: **{n_fdr}/{len(cfg)}**.")
    L.append(f"- Surviving Bonferroni at 5%: **{n_bonf}/{len(cfg)}**.")
    L.append(f"- Of the FDR-surviving ones, {len(pos)} are positive and {len(neg)} are "
             "**negative** (the model ranks engines backwards). A capability cannot "
             "change sign across configurations of the same model; an artefact can.")
    L.append(f"- Largest |rho| over any LLM configuration: **{cfg.spearman_rho.abs().max():.3f}** "
             f"(explains {100 * cfg.spearman_rho.abs().max() ** 2:.1f}% of rank variance), "
             f"against **{base_df[base_df.model == 'RandomForest'].spearman_rho.max():.3f}** "
             "for Random Forest on the same engines.")
    ci_cross = int(((cfg.rho_ci_lo < 0) & (cfg.rho_ci_hi > 0)).sum())
    L.append(f"- Bootstrap 95% CI includes zero in **{ci_cross}/{len(cfg)}** configurations.")

    (OUT / "p0_1b_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p0_1b_report.md'}")


if __name__ == "__main__":
    main()
