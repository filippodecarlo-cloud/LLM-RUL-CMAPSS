"""
P1.1 / P1.2 - Is the collapse a property of llama3.1:8b Q4, or of the approach?

The categorical collapse could be an artefact of this one model at this one
quantisation. This script holds everything else fixed and swaps the model.

Canonical configuration, identical to the P0 `repl` arm so the results are
directly comparable to the 5 existing runs of it:
    FD001, zero-shot, n=30, 100 test engines, T=0.1, shipped prompt template.

  P1.2  different models at similar size (mistral, qwen2.5, deepseek-r1, gemma)
  P1.1  same model, fp16 vs Q4_K_M (pass llama3.1:8b-instruct-fp16)

Two safeguards the original harness lacked:
  * reasoning-model output (<think> blocks) is stripped before parsing;
  * every response is stored raw, and a parse_ok flag records whether
    RUL_ESTIMATE was actually found - so a collapse can never be an artefact
    of the parser's fallback value.

Resumable: writes after every engine, skips finished ones on restart.

    python p1/p1_model_sweep.py --models mistral:7b qwen2.5:7b
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402

OUT = EXP / "results_p1"
OUT.mkdir(exist_ok=True)

DATASET, N_CYCLES, TEMPERATURE = "FD001", 30, 0.1
RUL_CAP = 125

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RUL_RE = re.compile(r"RUL_ESTIMATE:\s*(\d+)", re.IGNORECASE)


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def parse_response(text):
    """Return (rul, reasoning, parse_ok). Never silently invents a value."""
    clean = THINK_RE.sub(" ", text or "").strip()
    m = RUL_RE.search(clean)
    rul, ok = None, False
    if m:
        rul, ok = int(np.clip(int(m.group(1)), 0, RUL_CAP)), True
    else:
        nums = re.findall(r"\b(\d{1,3})\b", clean)
        cands = [int(n) for n in nums if 0 <= int(n) <= RUL_CAP]
        if cands:
            rul = cands[0]

    reasoning = ""
    mr = re.search(r"REASONING:\s*(.+)", clean, re.DOTALL | re.IGNORECASE)
    if mr:
        reasoning = " ".join(mr.group(1).split())
    return rul, reasoning, ok


def run_model(model, test_df, test_last, client, force=False):
    path = OUT / f"traces_p1_{DATASET}_zs_n{N_CYCLES}_{safe_name(model)}.json"
    done = {}
    if path.exists() and not force:
        done = {t["engine_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
        print(f"  [{model}] resuming, {len(done)} engines already done", flush=True)

    traces, t0 = [], time.time()
    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = int(row["engine_id"])
        if eid in done:
            traces.append(done[eid])
            continue

        g = test_df[test_df.engine_id == eid]
        prompt = ex.ZERO_SHOT_TEMPLATE.format(
            dataset=DATASET, n=N_CYCLES, sensor_block=ex._sensor_block(g))

        rec = None
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                raw = r.choices[0].message.content or ""
                rul, reasoning, ok = parse_response(raw)
                rec = {"engine_id": eid, "true_rul": float(row["RUL"]),
                       "pred_rul": rul, "reasoning": reasoning,
                       "parse_ok": ok, "raw": raw, "model": model}
                break
            except Exception as e:
                print(f"    [{model}] engine {eid} attempt {attempt + 1}: {e}", flush=True)
                time.sleep((attempt + 1) * 5)
        if rec is None:
            print(f"    [{model}] engine {eid} FAILED - skipped", flush=True)
            continue

        traces.append(rec)
        path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"    [{model}] {i + 1}/{len(test_last)} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{model}] done -> {path.name} ({len(traces)} engines, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    return traces


def summarise(model, traces, y_all):
    usable = [t for t in traces if t["pred_rul"] is not None]
    y = np.array([t["true_rul"] for t in usable], float)
    p = np.array([t["pred_rul"] for t in usable], float)
    vals, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    from scipy.stats import spearmanr
    rho = spearmanr(p, y) if len(vals) > 1 else None
    const = float(np.mean(y_all))
    return {
        "model": model, "n": len(usable),
        "parse_ok_pct": 100 * np.mean([t["parse_ok"] for t in traces]) if traces else np.nan,
        "unparsed": int(sum(1 for t in traces if t["pred_rul"] is None)),
        "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p), "NASA_Score": ex.nasa_score(y, p),
        "distinct_values": int(len(vals)),
        "modal_value": float(vals[counts.argmax()]),
        "modal_share_pct": 100 * counts.max() / len(p),
        "entropy_bits": float(-(q * np.log2(q)).sum()),
        "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)),
        "spearman_rho": float(rho.statistic) if rho else np.nan,
        "spearman_p": float(rho.pvalue) if rho else np.nan,
        "RMSE_const_trainmean": ex.rmse(y, np.full_like(y, const)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--engines", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg.N_CYCLES_IN_PROMPT = N_CYCLES
    train_df, test_df, test_last = ex.load_cmapss(DATASET)
    if args.engines:
        test_last = test_last.head(args.engines)
    y_all = train_df["RUL"].values

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", timeout=600)
    print(f"P1 model sweep - {DATASET}, zero-shot, n={N_CYCLES}, "
          f"{len(test_last)} engines, T={TEMPERATURE}", flush=True)

    rows = []
    for model in args.models:
        print(f"\n[model] {model}", flush=True)
        traces = run_model(model, test_df, test_last, client, force=args.force)
        if traces:
            rows.append(summarise(model, traces, y_all))
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in rows[-1].items()}, flush=True)

    if rows:
        df = pd.DataFrame(rows)
        out = OUT / "p1_model_summary.csv"
        if out.exists():
            prev = pd.read_csv(out)
            df = (pd.concat([prev[~prev.model.isin(df.model)], df])
                    .sort_values("model").reset_index(drop=True))
        df.to_csv(out, index=False)
        print("\n" + df.to_string(index=False))
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
