"""
The k x n few-shot grid, run with a correct example sampler.

The grid in the earlier release was not an ablation over k: its example pool was
built from the last cycle of each training engine, where the piecewise target is
zero by construction, so k=3 and k=5 received one example and k=10 received
three (see p1/p1_0_fewshot_bug.py). This script runs the same grid with examples
drawn properly, stratified across the three RUL bins, with labels anywhere in
0..125.

Running it means the paper can report a genuine ablation over k rather than
explaining the absence of one.

  k in {3, 5, 10} x n in {5, 15, 30}, FD001, 100 test engines, T=0.1.

Resumable: writes after every engine, skips completed ones on restart.

    python p1/p1_grid_fixed.py --k 5 --n 15
    python p1/p1_grid_fixed.py --all
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402

OUT = EXP / "results_p1"
OUT.mkdir(exist_ok=True)
DATASET, MODEL, TEMPERATURE, SEED = "FD001", "llama3.1:8b", 0.1, 1


def pick_stratified(train_df, k, n_cycles, seed=SEED):
    """k examples spread across the three RUL bins, drawn from real windows."""
    rng_state = int(seed)
    cand = (train_df.groupby("engine_id").cumcount() >= n_cycles - 1).values
    pool = train_df[cand][["engine_id", "RUL"]].reset_index(drop=True)
    pool = pool.assign(bin=pd.cut(pool["RUL"], bins=[0, 40, 80, 125],
                                  labels=[0, 1, 2], include_lowest=True).astype(int))
    per_bin, rest = k // 3, k % 3
    parts = []
    for b in range(3):
        take = per_bin + (1 if b < rest else 0)
        sub = pool[pool["bin"] == b]
        if len(sub):
            parts.append(sub.sample(n=min(take, len(sub)), random_state=rng_state + b))
    rows = pd.concat(parts).head(k)

    examples = []
    for _, r in rows.iterrows():
        eid = int(r.engine_id)
        g = train_df[train_df.engine_id == eid].reset_index(drop=True)
        hits = g.index[g["RUL"] == r.RUL].tolist()
        cut = max(h for h in hits if h >= n_cycles - 1)
        examples.append({"sensor_block": ex._sensor_block(g.iloc[:cut + 1]),
                         "true_rul": int(r.RUL), "engine_id": eid})
    return examples


def run(k, n, train_df, test_df, test_last, client, force=False):
    cfg.N_CYCLES_IN_PROMPT = n
    tag = f"k{k}_n{n}"
    path = OUT / f"traces_p1_grid_{DATASET}_fs_{tag}.json"
    examples = pick_stratified(train_df, k, n)
    print(f"  [{tag}] {len(examples)} examples, RULs "
          f"{[e['true_rul'] for e in examples]}", flush=True)

    done = {}
    if path.exists() and not force:
        done = {t["engine_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
        if len(done) >= len(test_last):
            print(f"  [{tag}] already complete, skipping", flush=True)
            return done_summary(tag, k, n, list(done.values()))
        print(f"  [{tag}] resuming from {len(done)}", flush=True)

    traces, t0 = [], time.time()
    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = int(row["engine_id"])
        if eid in done:
            traces.append(done[eid])
            continue
        g = test_df[test_df.engine_id == eid]
        prompt = ex.FEW_SHOT_TEMPLATE.format(
            dataset=DATASET, n=n,
            examples=ex._build_example_block(examples),
            sensor_block=ex._sensor_block(g))

        rec = None
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                rul, reasoning = ex._parse_response(r.choices[0].message.content or "")
                rec = {"engine_id": eid, "true_rul": float(row["RUL"]),
                       "pred_rul": rul, "reasoning": reasoning, "k": k, "n": n}
                break
            except Exception as e:
                print(f"    [{tag}] engine {eid} attempt {attempt + 1}: {e}", flush=True)
                time.sleep((attempt + 1) * 5)
        if rec is None:
            continue
        traces.append(rec)
        path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
        if (i + 1) % 25 == 0:
            el = (time.time() - t0) / 60
            print(f"    [{tag}] {i + 1}/{len(test_last)} ({el:.1f} min)", flush=True)

    path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{tag}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return done_summary(tag, k, n, traces)


def done_summary(tag, k, n, traces):
    y = np.array([t["true_rul"] for t in traces], float)
    p = np.array([t["pred_rul"] for t in traces], float)
    v, c = np.unique(p, return_counts=True)
    q = c / c.sum()
    rho = spearmanr(p, y) if len(v) > 1 else None
    return {"config": tag, "k": k, "n": n, "engines": len(p),
            "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
            "NASA_Score": ex.nasa_score(y, p),
            "distinct_values": int(len(v)),
            "modal_value": float(v[c.argmax()]),
            "modal_share_pct": 100 * c.max() / len(p),
            "entropy_bits": float(-(q * np.log2(q)).sum()),
            "pred_mean": float(p.mean()),
            "spearman_rho": float(rho.statistic) if rho else np.nan,
            "spearman_p": float(rho.pvalue) if rho else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--engines", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    train_df, test_df, test_last = ex.load_cmapss(DATASET)
    if args.engines:
        test_last = test_last.head(args.engines)
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)

    combos = ([(k, n) for k in (3, 5, 10) for n in (5, 15, 30)]
              if args.all else [(args.k, args.n)])
    print(f"Corrected few-shot grid: {len(combos)} configurations, "
          f"{len(test_last)} engines each", flush=True)

    rows = []
    out_csv = OUT / "p1_grid_fixed_summary.csv"
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
    for k, n in combos:
        print(f"\n[k={k}, n={n}]", flush=True)
        r = run(k, n, train_df, test_df, test_last, client, force=args.force)
        rows = [x for x in rows if x.get("config") != r["config"]] + [r]
        pd.DataFrame(rows).sort_values(["k", "n"]).to_csv(out_csv, index=False)
        print("   ", {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                      for kk, vv in r.items()}, flush=True)

    df = pd.DataFrame(rows).sort_values(["k", "n"])
    print("\n" + df.to_string(index=False))
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
