"""
P2.2 - Does the collapse survive multiple operating conditions?

FD002 (6 conditions, 1 fault mode) and FD004 (6 conditions, 2 fault modes) are
the hard half of CMAPSS, and reviewer critique D asked why they were left out.
One canonical configuration per dataset, matching the P0 `repl` arm:
zero-shot, n=30, T=0.1, shipped template, every test engine.

NORMALISATION. FD001/FD003 have a single operating condition, so the shipped
global MinMax scaler is harmless there. On FD002/FD004 a global scaler mixes six
regimes: the sensor value at a given wear level depends mostly on which regime
the engine is flying, and real degradation is buried under regime switching.
The standard remedy - and the one that gives the BASELINES their best shot, so
the comparison stays conservative with respect to our thesis - is to scale each
sensor within its operating-condition cluster. That is what this script does,
and the difference from FD001/FD003 has to be stated in the paper.

Baselines (RandomForest, XGBoost, repaired LSTM) and the no-skill constant are
computed on the identical split, so the LLM is never compared against a number
produced under different preprocessing.

Resumable. Usage:
    python p1/p2_2_multicondition.py --datasets FD002 FD004
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402
from p2_1_baselines import lstm_fixed, run_xgb  # noqa: E402

OUT = EXP / "results_p1"
OUT.mkdir(exist_ok=True)
DATA = EXP / "cmapss_data"

MODEL, TEMPERATURE, N_CYCLES, RUL_CAP = "llama3.1:8b", 0.1, 30, 125
N_REGIMES = 6
ALL_COLS = (["engine_id", "cycle", "setting1", "setting2", "setting3"] +
            [f"s{i}" for i in range(1, 22)])


def load_multicondition(dataset):
    """Like experiment.load_cmapss, but scaling per operating-condition cluster."""
    def _read(p):
        return pd.read_csv(p, sep=r"\s+", header=None, names=ALL_COLS,
                           engine="python").dropna(axis=1)

    train = _read(DATA / f"train_{dataset}.txt")
    test = _read(DATA / f"test_{dataset}.txt")
    rul = pd.read_csv(DATA / f"RUL_{dataset}.txt", sep=r"\s+", header=None,
                      names=["RUL"], engine="python").dropna(axis=1)

    maxc = train.groupby("engine_id")["cycle"].max().rename("max_cycle")
    train = train.join(maxc, on="engine_id")
    train["RUL"] = (train["max_cycle"] - train["cycle"]).clip(upper=RUL_CAP)
    train.drop(columns="max_cycle", inplace=True)

    # some sensor columns arrive as int64; scaled values must land in float
    train[cfg.SENSOR_NAMES] = train[cfg.SENSOR_NAMES].astype(float)
    test[cfg.SENSOR_NAMES] = test[cfg.SENSOR_NAMES].astype(float)

    settings = ["setting1", "setting2", "setting3"]
    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10).fit(train[settings])
    train["regime"] = km.predict(train[settings])
    test["regime"] = km.predict(test[settings])

    # one scaler per regime, fitted on train only
    for r in range(N_REGIMES):
        m_tr, m_te = train.regime == r, test.regime == r
        if not m_tr.any():
            continue
        sc = MinMaxScaler().fit(train.loc[m_tr, cfg.SENSOR_NAMES])
        train.loc[m_tr, cfg.SENSOR_NAMES] = sc.transform(train.loc[m_tr, cfg.SENSOR_NAMES])
        if m_te.any():
            test.loc[m_te, cfg.SENSOR_NAMES] = np.clip(
                sc.transform(test.loc[m_te, cfg.SENSOR_NAMES]), 0, 1)

    test_last = test.groupby("engine_id").last().reset_index()
    test_last["RUL"] = rul["RUL"].clip(upper=RUL_CAP).values
    return train, test, test_last


def describe(y, p):
    vals, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    degenerate = bool(p.max() - p.min() < 1e-3)
    rho = spearmanr(p, y) if (len(vals) > 1 and not degenerate) else None
    return {"RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
            "NASA_Score": ex.nasa_score(y, p),
            "spearman_rho": float(rho.statistic) if rho else np.nan,
            "spearman_p": float(rho.pvalue) if rho else np.nan,
            "distinct_values": int(len(vals)),
            "modal_value": float(vals[counts.argmax()]),
            "modal_share_pct": 100 * counts.max() / len(p),
            "entropy_bits": float(-(q * np.log2(q)).sum()),
            "pred_mean": float(p.mean())}


def run_llm(dataset, test_df, test_last, client, force=False):
    path = OUT / f"traces_p2_{dataset}_zs_n{N_CYCLES}.json"
    done = {}
    if path.exists() and not force:
        done = {t["engine_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
        print(f"  [{dataset}] resuming, {len(done)} done", flush=True)

    traces, t0 = [], time.time()
    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = int(row["engine_id"])
        if eid in done:
            traces.append(done[eid])
            continue
        g = test_df[test_df.engine_id == eid]
        prompt = ex.ZERO_SHOT_TEMPLATE.format(
            dataset=dataset, n=N_CYCLES, sensor_block=ex._sensor_block(g))

        rec = None
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                rul, reasoning = ex._parse_response(r.choices[0].message.content or "")
                rec = {"engine_id": eid, "true_rul": float(row["RUL"]),
                       "pred_rul": rul, "reasoning": reasoning}
                break
            except Exception as e:
                print(f"    [{dataset}] engine {eid} attempt {attempt+1}: {e}", flush=True)
                time.sleep((attempt + 1) * 5)
        if rec is None:
            continue
        traces.append(rec)
        path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
        if (i + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            rate = el / max(1, len(traces) - len(done))
            print(f"    [{dataset}] {i+1}/{len(test_last)} ({el:.1f} min, "
                  f"eta {rate * (len(test_last) - i - 1):.0f} min)", flush=True)

    path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{dataset}] LLM done ({(time.time()-t0)/60:.1f} min)", flush=True)
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["FD002", "FD004"])
    ap.add_argument("--engines", type=int, default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg.N_CYCLES_IN_PROMPT = N_CYCLES
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
    rows = []

    for ds in args.datasets:
        print(f"\n=== {ds} ===", flush=True)
        train_df, test_df, test_last = load_multicondition(ds)
        if args.engines:
            test_last = test_last.head(args.engines)
        y = test_last["RUL"].values.astype(float)
        print(f"  {len(test_last)} test engines, "
              f"{train_df.engine_id.nunique()} train engines, "
              f"{N_REGIMES} operating regimes", flush=True)

        traces = run_llm(ds, test_df, test_last, client, force=args.force)
        usable = [t for t in traces if t["pred_rul"] is not None]
        yy = np.array([t["true_rul"] for t in usable], float)
        pp = np.array([t["pred_rul"] for t in usable], float)
        rows.append({"dataset": ds, "model": "llama3.1:8b zero-shot n=30",
                     "n": len(pp), **describe(yy, pp)})

        c = float(train_df["RUL"].mean())
        rows.append({"dataset": ds, "model": "constant = train mean", "n": len(y),
                     **describe(y, np.full_like(y, c))})

        if not args.skip_baselines:
            print(f"  [{ds}] RandomForest", flush=True)
            rf, _ = ex.run_rf(train_df, test_df, test_last)
            rows.append({"dataset": ds, "model": "RandomForest", "n": len(y),
                         **describe(y, np.asarray(rf, float))})
            print(f"  [{ds}] XGBoost", flush=True)
            rows.append({"dataset": ds, "model": "XGBoost", "n": len(y),
                         **describe(y, run_xgb(train_df, test_df, test_last))})
            print(f"  [{ds}] LSTM (repaired)", flush=True)
            lp, _ = lstm_fixed(train_df, test_df, test_last)
            rows.append({"dataset": ds, "model": "LSTM (repaired)", "n": len(y),
                         **describe(y, lp)})

        pd.DataFrame(rows).to_csv(OUT / "p2_2_summary.csv", index=False)

    df = pd.DataFrame(rows)
    L = ["# P2.2 - Multiple operating conditions (FD002, FD004)\n"]
    L.append("Zero-shot, n=30, T=0.1, shipped template, every test engine. Sensors scaled "
             "**within each of the 6 operating-condition clusters** (k-means on the three "
             "setting channels, fitted on train) rather than globally - on FD001/FD003 a "
             "single condition made that distinction irrelevant, here it is essential and "
             "it favours the baselines.\n")
    for ds in df.dataset.unique():
        L.append(f"\n## {ds}\n")
        L.append("| Model | n | RMSE | MAE | NASA | rho | distinct | modal (share) | entropy |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in df[df.dataset == ds].iterrows():
            rho = "n/a" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.3f}"
            L.append(f"| {r.model} | {r.n:.0f} | {r.RMSE:.2f} | {r.MAE:.2f} | "
                     f"{r.NASA_Score:,.0f} | {rho} | {r.distinct_values:.0f} | "
                     f"{r.modal_value:.0f} ({r.modal_share_pct:.0f}%) | {r.entropy_bits:.2f} |")
    (OUT / "p2_2_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p2_2_report.md'}")


if __name__ == "__main__":
    main()
