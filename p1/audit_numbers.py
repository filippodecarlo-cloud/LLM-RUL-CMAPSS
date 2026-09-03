"""
Audit: does every number in the manuscript match the data?

Extracts numeric claims from paper_v14_draft.md and checks them against values
recomputed from the result files. Catches the failure mode that matters most in
a paper like this one: a number that was right when it was written, and stale
after the experiment that produced it was rerun.

Run it before every submission, and after any new experiment.

    python p1/audit_numbers.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"
PAPER = EXP.parent / "paper_v14_draft.md"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def facts():
    """Recompute every checkable quantity straight from the data."""
    f = {}

    base = pd.read_csv(P0 / "p0_1_noskill_baselines.csv")
    b1 = base[base.dataset == "FD001"]
    f["FD001 train-mean constant RMSE"] = float(
        b1[b1.predictor.str.contains("TRAIN mean")].RMSE.iloc[0])
    f["FD001 NASA-optimal constant NASA"] = float(
        b1[b1.predictor.str.contains("NASA-optimal")].NASA_Score.iloc[0])
    f["FD001 random-uniform RMSE"] = float(b1[b1.predictor.str.contains("random")].RMSE.iloc[0])

    cfg = pd.read_csv(P0 / "p0_1_llm_configs.csv")
    c1 = cfg[cfg.dataset == "FD001"]
    f["best LLM RMSE (FD001)"] = float(c1.RMSE.min())
    f["best LLM NASA (FD001)"] = float(c1.NASA_Score.min())
    f["configs beating the constant"] = int(cfg.beats_trainmean_RMSE.sum())
    f["total configs"] = int(len(cfg))

    bs = pd.read_csv(P1 / "p2_1_baseline_summary.csv")
    for ds in ("FD001", "FD003"):
        for m, lab in (("RandomForest", "RF"), ("XGBoost", "XGB"),
                       ("LSTM_original", "LSTM orig"), ("LSTM_fixed", "LSTM fixed")):
            r = bs[(bs.dataset == ds) & (bs.model == m)]
            if len(r):
                f[f"{ds} {lab} RMSE"] = float(r.RMSE.iloc[0])
                if not pd.isna(r.spearman_rho.iloc[0]):
                    f[f"{ds} {lab} rho"] = float(r.spearman_rho.iloc[0])

    # model sweep
    runs = {"llama Q4": EXP / "results" / "traces_FD001_zero_shot_k0_n30.json"}
    for p in P1.glob("traces_p1_FD001_zs_n30_*.json"):
        runs[p.stem.replace("traces_p1_FD001_zs_n30_", "")] = p
    for lab, p in runs.items():
        tr = json.loads(p.read_text(encoding="utf-8"))
        pr = np.array([t["pred_rul"] for t in tr if t.get("pred_rul") is not None], float)
        y = np.array([t["true_rul"] for t in tr if t.get("pred_rul") is not None], float)
        v, c = np.unique(pr, return_counts=True)
        q = c / c.sum()
        f[f"{lab}: distinct"] = int(len(v))
        f[f"{lab}: modal share %"] = round(100 * c.max() / len(pr), 1)
        f[f"{lab}: entropy bits"] = round(float(-(q * np.log2(q)).sum()), 2)
        f[f"{lab}: RMSE"] = rmse(y, pr)

    # multi-condition
    m = pd.read_csv(P1 / "p2_2_summary.csv")
    for _, r in m.iterrows():
        if "llama" in str(r.model):
            f[f"{r.dataset} LLM RMSE"] = float(r.RMSE)
            f[f"{r.dataset} LLM modal share %"] = round(float(r.modal_share_pct), 1)

    # faithfulness
    cl = pd.read_csv(P0 / "p0_2a_claims.csv")
    d = cl[cl.claimed.isin(["increase", "decrease"])]
    f["sensor-trace opportunities"] = int(len(cl))
    f["directional claims"] = int(len(d))
    f["textbook agreement %"] = round(100 * (d.claimed == d.expected_textbook).mean(), 1)
    f["input faithfulness %"] = round(100 * (d.claimed == d.actual_in_window).mean(), 1)
    f["base rate %"] = round(100 * (cl.expected_textbook == cl.actual_in_window).mean(), 1)
    f["cued agreement %"] = round(100 * (d[d.cued].claimed == d[d.cued].expected_textbook).mean(), 1)
    f["uncued agreement %"] = round(
        100 * (d[~d.cued].claimed == d[~d.cued].expected_textbook).mean(), 1)

    # control arms
    a = pd.read_csv(P0 / "p0_23_arm_summary.csv").set_index("arm")
    for arm in a.index:
        f[f"arm {arm} RMSE"] = float(a.loc[arm, "RMSE"])
        f[f"arm {arm} modal share %"] = round(float(a.loc[arm, "modal_share_pct"]), 1)

    # reversal test
    cl2 = pd.read_csv(P0 / "p0_23_claims.csv")
    piv = cl2[cl2.arm.isin(["repl", "revdata"])].pivot_table(
        index=["engine_id", "sensor"], columns="arm",
        values=["claimed", "actual_shown"], aggfunc="first")
    piv.columns = [f"{x}_{y}" for x, y in piv.columns]
    piv = piv.dropna()
    both = piv[piv.claimed_repl.isin(["increase", "decrease"]) &
               piv.claimed_revdata.isin(["increase", "decrease"])]
    flip = both[both.actual_shown_repl != both.actual_shown_revdata]
    chg = flip[flip.claimed_repl != flip.claimed_revdata]
    f["reversed pairs"] = int(len(flip))
    f["claims that changed"] = int(len(chg))
    f["claims changed %"] = round(100 * len(chg) / max(len(flip), 1), 1)

    # anchoring
    import importlib.util
    import config as cfg2
    spec = importlib.util.spec_from_file_location("fs", HERE / "p1_34_fewshot_sampling.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    cfg2.N_CYCLES_IN_PROMPT = fs.N_CYCLES
    train_df, _, _ = fs.ex.load_cmapss(fs.DATASET)
    an = pd.read_csv(P1 / "p1_34_analysis.csv")
    an = an[(an.n >= 100) & (an.condition != "published")]
    xs, ys = [], []
    for _, r in an.iterrows():
        cond, seed = r.run.rsplit("_seed", 1)
        _, exs = fs.pick_examples(train_df, cond, int(seed))
        xs.append(float(np.mean([e["true_rul"] for e in exs])))
        ys.append(float(r.pred_mean))
    pr_ = pearsonr(xs, ys)
    f["anchoring r"] = round(float(pr_.statistic), 3)
    f["anchoring R2 %"] = round(100 * float(pr_.statistic) ** 2, 0)
    f["example sets"] = len(xs)

    # rank correlation family
    rk = pd.read_csv(P0 / "p0_1b_rank_significance.csv")
    f["nominally significant"] = int((rk.spearman_p < 0.05).sum())
    f["FDR-surviving"] = int(rk.sig_fdr_5pct.sum())
    f["max |rho| LLM"] = round(float(rk.spearman_rho.abs().max()), 3)

    # total inferences
    tot = 0
    for pat in ("results/traces_*.json", "results_p0/traces_p0_*.json",
                "results_p1/traces_p*.json"):
        for p in EXP.glob(pat):
            tot += len(json.loads(p.read_text(encoding="utf-8")))
    f["total inferences"] = tot
    return f


# Numbers as they appear in the manuscript, paired with the fact they must match.
CHECKS = [
    ("41.94", "FD001 train-mean constant RMSE"), ("45.60", "best LLM RMSE (FD001)"),
    ("18,572", "best LLM NASA (FD001)"), ("5,506", "FD001 NASA-optimal constant NASA"),
    ("55.02", "FD001 random-uniform RMSE"),
    ("17.96", "FD001 RF RMSE"), ("15.95", "FD001 XGB RMSE"),
    ("43.89", "FD001 LSTM orig RMSE"), ("13.58", "FD001 LSTM fixed RMSE"),
    ("13.81", "FD003 LSTM fixed RMSE"), ("0.889", "FD001 LSTM fixed rho"),
    ("0.813", "FD001 RF rho"),
    ("46.88", "mistral_7b: RMSE"), ("71.69", "qwen2.5_7b: RMSE"),
    ("67.31", "deepseek-r1_7b: RMSE"), ("3.52", "deepseek-r1_7b: entropy bits"),
    ("19", "deepseek-r1_7b: distinct"),
    ("63.20", "FD002 LLM RMSE"), ("66.56", "FD004 LLM RMSE"),
    ("18,900", "sensor-trace opportunities"), ("3,816", "directional claims"),
    ("91.3", "textbook agreement %"), ("42.7", "input faithfulness %"),
    ("46.7", "base rate %"), ("94.5", "cued agreement %"), ("51.4", "uncued agreement %"),
    ("61.57", "arm repl RMSE"), ("59.13", "arm nocue RMSE"),
    ("62.32", "arm invcue RMSE"), ("61.21", "arm revdata RMSE"),
    ("253", "reversed pairs"), ("2.4", "claims changed %"),
    ("0.941", "anchoring r"), ("89", "anchoring R2 %"),
    ("4,657", "total inferences"),
]


def main():
    f = facts()
    text = PAPER.read_text(encoding="utf-8")

    print(f"Audit of {PAPER.name} against the result files\n" + "=" * 62)
    bad, missing, ok = [], [], 0
    for shown, key in CHECKS:
        if key not in f:
            missing.append((shown, key))
            continue
        actual = f[key]
        as_written = float(shown.replace(",", ""))
        match = abs(as_written - float(actual)) < max(0.051, abs(float(actual)) * 0.002)
        in_paper = shown in text
        if not in_paper:
            missing.append((shown, key + "  [NOT FOUND IN PAPER]"))
        elif match:
            ok += 1
        else:
            bad.append((shown, key, actual))

    if bad:
        print("\nMISMATCHES (paper says X, data says Y):")
        for shown, key, actual in bad:
            print(f"  {key:38s} paper: {shown:>10s}   data: {actual}")
    if missing:
        print("\nNOT CHECKED (value absent from the paper, or fact unavailable):")
        for shown, key in missing:
            print(f"  {key:38s} expected to find: {shown}")

    print(f"\n{ok} checked and matching, {len(bad)} mismatched, {len(missing)} not checked.")
    if not bad:
        print("\nEvery number checked in the manuscript matches the data.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
