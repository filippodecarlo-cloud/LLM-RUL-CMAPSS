"""
P1.3 / P1.4 - Does the few-shot example set explain the anchors, and how much do
the results move when it is resampled?

P1.0 established that the shipped few-shot builder never stratifies: its pool is
the last cycle of every training engine, so every example is labelled RUL = 0 and
"k" examples are really 1 (k=3, k=5) or 3 (k=10). That makes two questions
testable rather than rhetorical:

  P1.3
        if the low anchors come from being shown only end-of-life engines, then
        supplying genuinely stratified examples should move them.
  P1.4
        results from one arbitrary example set were never robustness-checked.

Three conditions, everything else fixed (FD001, k=5, n=5, 100 engines, T=0.1,
shipped template):

  asis_seed1        the shipped selection reproduced verbatim - 1 example, RUL 0.
                    Deterministic, so one run is enough.
  strat_seed<S>     genuine stratification: 5 examples spread across the three
                    RUL bins, labels anywhere in 0..125.
  random_seed<S>    5 windows drawn uniformly at random, no stratification.

Unlike the shipped code, an example here is a WINDOW ending at an arbitrary
cycle, so its label can be any RUL - not only 0.

`asis` vs `strat` isolates the defect. `strat` vs `random` isolates
stratification proper. Seeds within a condition give the sampling variability
the paper never reported.

Resumable. Usage:
    python p1/p1_34_fewshot_sampling.py --seeds 1 2 3
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

DATASET, K, N_CYCLES, TEMPERATURE = "FD001", 5, 5, 0.1
MODEL = "llama3.1:8b"


def _block_ending_at(g, cut_idx):
    """Sensor block for the N_CYCLES ending at row cut_idx of engine frame g."""
    return ex._sensor_block(g.iloc[:cut_idx + 1])


def pick_examples(train_df, condition, seed):
    """
    Return (description, examples). An example is a window plus the RUL at its
    last cycle - so unlike the published code, the label can be any value in
    0..125, not only 0.

    asis    reproduce the shipped selection verbatim (see p1_0_fewshot_bug.py):
            last cycle of the first engines, every label RUL = 0.
    strat   genuine stratification: examples drawn from all three RUL bins.
    random  windows drawn uniformly at random, no stratification.
    """
    rng = np.random.default_rng(seed)

    if condition == "asis":
        pool = train_df.groupby("engine_id").last().reset_index()
        pool["bin"] = pd.cut(pool["RUL"], bins=[0, 40, 80, 125],
                             labels=[0, 1, 2], include_lowest=True).astype(int)
        per_bin = max(1, K // 3)
        ids = []
        for b in range(3):
            ids.extend(pool[pool["bin"] == b]["engine_id"].tolist()[:per_bin])
        ids = ids[:K]
        examples = []
        for eid in ids:
            g = train_df[train_df.engine_id == eid].reset_index(drop=True)
            examples.append({"sensor_block": ex._sensor_block(g),
                             "true_rul": int(g["RUL"].iloc[-1]),
                             "engine_id": int(eid)})
        return "shipped selection (verbatim)", examples

    # Candidate cut points: any cycle with at least N_CYCLES of history.
    cand = (train_df.groupby("engine_id").cumcount() >= N_CYCLES - 1).values
    pool = train_df[cand][["engine_id", "RUL"]].reset_index(drop=True)

    if condition == "random":
        rows = pool.sample(n=K, random_state=int(seed))
        desc = "uniform random windows, no stratification"
    else:
        pool = pool.assign(bin=pd.cut(pool["RUL"], bins=[0, 40, 80, 125],
                                      labels=[0, 1, 2],
                                      include_lowest=True).astype(int))
        per_bin, rest = K // 3, K % 3
        parts = []
        for b in range(3):
            take = per_bin + (1 if b < rest else 0)
            sub = pool[pool["bin"] == b]
            if len(sub):
                parts.append(sub.sample(n=min(take, len(sub)), random_state=int(seed) + b))
        rows = pd.concat(parts).head(K)
        desc = "true stratification across the three RUL bins"

    examples = []
    for _, r in rows.iterrows():
        eid = int(r.engine_id)
        g = train_df[train_df.engine_id == eid].reset_index(drop=True)
        hits = g.index[g["RUL"] == r.RUL].tolist()
        cut = max(h for h in hits if h >= N_CYCLES - 1)
        examples.append({"sensor_block": _block_ending_at(g, cut),
                         "true_rul": int(r.RUL), "engine_id": eid})
    return desc, examples


def run_condition(condition, seed, train_df, test_df, test_last, client, force=False):
    tag = f"{condition}_seed{seed}"
    path = OUT / f"traces_p1_{DATASET}_fs_k{K}_n{N_CYCLES}_{tag}.json"
    desc, examples = pick_examples(train_df, condition, seed)
    ids = [e["engine_id"] for e in examples]
    ex_ruls = [e["true_rul"] for e in examples]
    print(f"  [{tag}] {desc}: {len(examples)} examples, engines {ids}, "
          f"RULs {ex_ruls}", flush=True)

    done = {}
    if path.exists() and not force:
        done = {t["engine_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
        print(f"  [{tag}] resuming, {len(done)} done", flush=True)

    traces, t0 = [], time.time()
    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = int(row["engine_id"])
        if eid in done:
            traces.append(done[eid])
            continue

        g = test_df[test_df.engine_id == eid]
        prompt = ex.FEW_SHOT_TEMPLATE.format(
            dataset=DATASET, n=N_CYCLES,
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
                       "pred_rul": rul, "reasoning": reasoning,
                       "condition": condition, "seed": seed}
                break
            except Exception as e:
                print(f"    [{tag}] engine {eid} attempt {attempt + 1}: {e}", flush=True)
                time.sleep((attempt + 1) * 5)
        if rec is None:
            continue

        traces.append(rec)
        path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
        if (i + 1) % 20 == 0:
            print(f"    [{tag}] {i + 1}/{len(test_last)} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    y = np.array([t["true_rul"] for t in traces], float)
    p = np.array([t["pred_rul"] for t in traces], float)
    vals, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    rho = spearmanr(p, y) if len(vals) > 1 else None
    print(f"  [{tag}] done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"condition": condition, "seed": seed, "n": len(p),
            "n_examples": len(examples), "example_engines": str(ids),
            "example_ruls": str(ex_ruls),
            "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
            "NASA_Score": ex.nasa_score(y, p),
            "distinct_values": int(len(vals)),
            "modal_value": float(vals[counts.argmax()]),
            "modal_share_pct": 100 * counts.max() / len(p),
            "entropy_bits": float(-(q * np.log2(q)).sum()),
            "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)),
            "spearman_rho": float(rho.statistic) if rho else np.nan,
            "spearman_p": float(rho.pvalue) if rho else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", default=["asis", "strat", "random"],
                    choices=["asis", "strat", "random"])
    ap.add_argument("--engines", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg.N_CYCLES_IN_PROMPT = N_CYCLES
    train_df, test_df, test_last = ex.load_cmapss(DATASET)
    if args.engines:
        test_last = test_last.head(args.engines)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
    print(f"P1.3/P1.4 few-shot sampling - {DATASET}, k={K}, n={N_CYCLES}, "
          f"{len(test_last)} engines, model={MODEL}", flush=True)

    rows = []
    for condition in args.conditions:
        for seed in args.seeds:
            print(f"\n[{condition} seed {seed}]", flush=True)
            rows.append(run_condition(condition, seed, train_df, test_df,
                                      test_last, client, force=args.force))
            pd.DataFrame(rows).to_csv(OUT / "p1_34_summary.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n" + df.drop(columns=["example_engines"]).to_string(index=False))
    print(f"\n[saved] {OUT / 'p1_34_summary.csv'}")


if __name__ == "__main__":
    main()
