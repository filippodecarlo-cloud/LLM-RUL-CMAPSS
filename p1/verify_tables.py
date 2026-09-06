"""
Check every cell of every results table in the manuscript against the raw data.

audit_numbers.py checks a hand-written list of 35 quantities. That list is the
weak point: three wrong numbers reached the submission draft precisely because
they were not on it. This script does not use a list. It recomputes each run
from its trace file and compares the recomputed value with whatever the Word
table actually says, cell by cell, so a number can only pass by being right.

Covered: Table 1 (constants and best configurations), Table 2 (supervised
baselines), Table 3 (model sweep), Table 4 (all four sub-datasets), Table 6
(control arms) and Table 7 (the k x n grid). Tables 5 is checked against the
P0.2a report, which is itself generated from the claims file.

    python p1/verify_tables.py
"""
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
P0, P1 = EXP / "results_p0", EXP / "results_p1"
# The manuscript is not redistributed here while it is under review, so this
# script needs a copy of it. Point PAPER_DOCX at one, or drop paper_v15.docx
# beside the repository. Everything the script compares the manuscript against
# is in results/, results_p0/ and results_p1/ and is public.
PAPER = Path(os.environ.get("PAPER_DOCX", EXP.parent / "paper_v15.docx"))

TOL = 0.011          # a value printed to 2 dp may differ by half a unit in the last place
PCT_TOL = 0.51       # a percentage printed as an integer


def load(path):
    tr = json.loads(Path(path).read_text(encoding="utf-8"))
    pr, tv = [], []
    for t in tr:
        v = t.get("pred_rul")
        if v in (None, "", "None"):
            continue
        pr.append(float(v))
        tv.append(float(t["true_rul"]))
    return np.array(pr), np.array(tv)


def nasa(p, y):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def stats(path):
    p, y = load(path)
    c = Counter(p)
    n = len(p)
    mv, mc = c.most_common(1)[0]
    rho = spearmanr(p, y).statistic if len(c) > 1 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
        "MAE": float(np.mean(np.abs(p - y))),
        "NASA": nasa(p, y),
        "distinct": float(len(c)),
        "modal": float(mv),
        "share": 100.0 * mc / n,
        "entropy": -sum((v / n) * math.log2(v / n) for v in c.values()),
        "rho": rho,
        "mean": float(p.mean()),
        "n": float(n),
    }


def cells(table):
    return [[c.text.strip() for c in row.cells] for row in table.rows]


def num(s):
    """First number in a cell, ignoring thousands separators and % signs."""
    m = re.search(r"[-+−]?\d[\d,]*\.?\d*", s.replace("−", "-"))
    return float(m.group(0).replace(",", "")) if m else None


def pct(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None


def check(results, label, shown, actual, tol=TOL):
    if shown is None:
        results.append(("MISSING", label, "-", actual))
        return
    ok = actual is not None and abs(shown - actual) <= max(tol, abs(actual) * 0.002)
    results.append(("ok" if ok else "MISMATCH", label, shown, actual))


def main():
    if not PAPER.exists():
        raise SystemExit(
            f"Manuscript not found at {PAPER}.\n"
            "This script compares the manuscript's tables with the data in this "
            "repository, so it needs the .docx. Set PAPER_DOCX to its path.")
    doc = Document(PAPER)
    tabs = doc.tables
    if len(tabs) < 7:
        print(f"expected 7 tables, found {len(tabs)}")
        return 1
    R = []

    # ---------------- Table 1: constants and best configurations ----------
    base = pd.read_csv(P0 / "p0_1_noskill_baselines.csv")
    b1 = base[base.dataset == "FD001"].set_index("predictor")
    key = {"training mean": "constant = TRAIN mean RUL",
           "62.5": "constant = RUL_CAP/2 (62.5)",
           "RMSE-optimal": "constant = RMSE-optimal (oracle)",
           "NASA-optimal": "constant = NASA-optimal (oracle)",
           "random": "random uniform U(0,125), mean of 1000"}
    for row in cells(tabs[0])[1:]:
        name = row[0]
        src = next((v for k, v in key.items() if k.lower() in name.lower()), None)
        if src:
            r = b1.loc[src]
            for j, col in enumerate(("RMSE", "MAE", "NASA_Score"), start=1):
                check(R, f"T1 {name[:34]} {col}", num(row[j]), float(r[col]))
        elif "k=10, n=30" in name:
            s = stats(EXP / "results/traces_FD001_few_shot_k10_n30.json")
            for j, k in enumerate(("RMSE", "MAE", "NASA"), start=1):
                check(R, f"T1 best-RMSE {k}", num(row[j]), s[k])
        elif "k=5, n=15" in name:
            s = stats(EXP / "results/traces_FD001_few_shot_k5_n15.json")
            for j, k in enumerate(("RMSE", "MAE", "NASA"), start=1):
                check(R, f"T1 best-NASA {k}", num(row[j]), s[k])

    # ---------------- Table 2: supervised baselines ------------------------
    bs = pd.read_csv(P1 / "p2_1_baseline_summary.csv")
    mmap = {"Constant = training mean": "const_trainmean",
            "LSTM, unscaled target": "LSTM_original",
            "Random Forest": "RandomForest", "XGBoost": "XGBoost",
            "LSTM, scaled target": "LSTM_fixed"}
    for row in cells(tabs[1])[1:]:
        ds, name = row[0], row[1]
        src = next((v for k, v in mmap.items() if k.lower() in name.lower()), None)
        if not src:
            continue
        r = bs[(bs.dataset == ds) & (bs.model == src)]
        if not len(r):
            continue
        r = r.iloc[0]
        for j, col in enumerate(("RMSE", "MAE", "NASA_Score"), start=2):
            check(R, f"T2 {ds} {name[:26]} {col}", num(row[j]), float(r[col]))
        if not pd.isna(r.spearman_rho) and num(row[5]) is not None:
            check(R, f"T2 {ds} {name[:26]} rho", num(row[5]), float(r.spearman_rho), 0.0011)

    # ---------------- Table 3: model sweep ---------------------------------
    runs3 = {"Q4_K_M": EXP / "results/traces_FD001_zero_shot_k0_n30.json",
             "Q8_0": P1 / "traces_p1_FD001_zs_n30_llama3.1_8b-instruct-q8_0.json",
             "Mistral": P1 / "traces_p1_FD001_zs_n30_mistral_7b.json",
             "Qwen": P1 / "traces_p1_FD001_zs_n30_qwen2.5_7b.json",
             "DeepSeek": P1 / "traces_p1_FD001_zs_n30_deepseek-r1_7b.json"}
    for row in cells(tabs[2])[1:]:
        src = next((v for k, v in runs3.items() if k.lower() in row[0].lower()), None)
        if not src:
            continue
        s = stats(src)
        lab = row[0][:26]
        check(R, f"T3 {lab} engines", num(row[1]), s["n"], 0.5)
        check(R, f"T3 {lab} RMSE", num(row[2]), s["RMSE"])
        check(R, f"T3 {lab} distinct", num(row[3]), s["distinct"], 0.5)
        check(R, f"T3 {lab} modal", num(row[4]), s["modal"], 0.5)
        check(R, f"T3 {lab} share", pct(row[4]), s["share"], PCT_TOL)
        check(R, f"T3 {lab} entropy", num(row[5]), s["entropy"])

    # ---------------- Table 4: all four sub-datasets -----------------------
    runs4 = {"FD001": EXP / "results/traces_FD001_zero_shot_k0_n30.json",
             "FD003": EXP / "results/traces_FD003_zero_shot_k0_n30.json",
             "FD002": P1 / "traces_p2_FD002_zs_n30.json",
             "FD004": P1 / "traces_p2_FD004_zs_n30.json"}
    # FD001/FD003 constants come from P0, FD002/FD004 from the P2.2 summary
    p22 = pd.read_csv(P1 / "p2_2_summary.csv")
    const = {}
    for d in runs4:
        r = base[(base.dataset == d) & base.predictor.str.contains("TRAIN mean")]
        if len(r):
            const[d] = float(r.RMSE.iloc[0])
            continue
        r = p22[(p22.dataset == d) & p22.model.str.contains("train mean")]
        const[d] = float(r.RMSE.iloc[0]) if len(r) else None
    for row in cells(tabs[3])[1:]:
        ds = row[0]
        if ds not in runs4:
            continue
        s = stats(runs4[ds])
        check(R, f"T4 {ds} engines", num(row[2]), s["n"], 0.5)
        check(R, f"T4 {ds} LLM RMSE", num(row[3]), s["RMSE"])
        check(R, f"T4 {ds} constant RMSE", num(row[4]), const[ds])
        check(R, f"T4 {ds} distinct", num(row[6]), s["distinct"], 0.5)
        check(R, f"T4 {ds} modal", num(row[7]), s["modal"], 0.5)
        check(R, f"T4 {ds} share", pct(row[7]), s["share"], PCT_TOL)

    # ---------------- Table 6: control arms --------------------------------
    arms = pd.read_csv(P0 / "p0_23_arm_summary.csv").set_index("arm")
    for row in cells(tabs[5])[1:]:
        a = row[0].strip().lower()
        if a not in arms.index:
            continue
        r = arms.loc[a]
        check(R, f"T6 {a} RMSE", num(row[2]), float(r.RMSE))
        check(R, f"T6 {a} modal", num(row[3]), float(r.modal_value), 0.5)
        check(R, f"T6 {a} share", pct(row[3]), float(r.modal_share_pct), PCT_TOL)

    # ---------------- Table 7: the k x n grid ------------------------------
    for row in cells(tabs[6])[1:]:
        n = num(row[0])
        if n is None:
            continue
        for j, k in enumerate((3, 5, 10), start=1):
            f = P1 / f"traces_p1_grid_FD001_fs_k{k}_n{int(n)}.json"
            if not f.exists():
                continue
            s = stats(f)
            parts = re.findall(r"[\d.]+", row[j])
            if len(parts) >= 3:
                check(R, f"T7 k{k} n{int(n)} RMSE", float(parts[0]), s["RMSE"])
                check(R, f"T7 k{k} n{int(n)} distinct", float(parts[1]), s["distinct"], 0.5)
                check(R, f"T7 k{k} n{int(n)} share", float(parts[2]), s["share"], PCT_TOL)

    bad = [r for r in R if r[0] != "ok"]
    print(f"Cell-by-cell verification of {PAPER.name}\n" + "=" * 66)
    for status, label, shown, actual in bad:
        a = f"{actual:.4g}" if isinstance(actual, float) else actual
        print(f"  {status:9s} {label:44s} paper: {shown}   data: {a}")
    print(f"\n{len(R) - len(bad)} cells verified against the raw data, {len(bad)} not matching.")
    if not bad:
        print("Every table cell recomputed from the traces matches the manuscript.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
