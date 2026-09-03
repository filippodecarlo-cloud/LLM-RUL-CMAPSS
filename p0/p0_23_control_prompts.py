"""
P0.2b / P0.3 - Control prompts and the inverted-trend faithfulness test.

Paired design. Every arm sees the SAME 100 FD001 test engines, the same model
(llama3.1:8b, T=0.1), the same zero-shot template and the same n=30 window.
Exactly one thing changes per arm, so any difference is attributable to it.

  repl     original prompt, real data.  Replication control: must land inside
           the band of the four existing runs (RMSE 61.5-62.8, modal value 23
           at 80-85%). Validates that this script reproduces experiment.py.

  nocue    P0.2b. The INSTRUCTIONS block that states which sensors rise and
           which fall is replaced by a neutral instruction. If the reported
           91% "explanation accuracy" is prompt leakage, it collapses here.

  invcue   The cue is reversed: the prompt asserts the opposite physics.
           If the explanations follow the prompt rather than the data, the
           claimed directions flip with it.

  revdata  P0.3. Original prompt, but each engine's window is reversed in
           time. Same numbers, same multiset, every trend sign flipped.
           A model reading the trends must change its directional claims;
           a model reciting a template will not.

Resumable: each arm writes after every engine, and rerunning skips finished
engines. Usage:
    python p0/p0_23_control_prompts.py --arms repl nocue invcue revdata
"""
import argparse
import json
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

OUT = EXP / "results_p0"
OUT.mkdir(exist_ok=True)

MODEL = "llama3.1:8b"
TEMPERATURE = 0.1
DATASET = "FD001"
N_CYCLES = 30

# The cue sentence in the shipped zero-shot template, verbatim.
ORIGINAL_CUE = (
    "Analyse the sensor trends. Higher values for degradation-related sensors "
    "(e.g., s11, s12, s15)\nand lower values for efficiency-related sensors "
    "(e.g., s9, s14) typically indicate advanced wear.\nIncreasing variance or "
    "monotonic drift also suggests approaching failure."
)

NEUTRAL_CUE = (
    "Analyse the sensor readings above and estimate the remaining useful life."
)

INVERTED_CUE = (
    "Analyse the sensor trends. Lower values for degradation-related sensors "
    "(e.g., s11, s12, s15)\nand higher values for efficiency-related sensors "
    "(e.g., s9, s14) typically indicate advanced wear.\nDecreasing variance or "
    "flat readings also suggests approaching failure."
)

ARMS = {
    "repl":    {"cue": ORIGINAL_CUE, "reverse_window": False},
    "nocue":   {"cue": NEUTRAL_CUE,  "reverse_window": False},
    "invcue":  {"cue": INVERTED_CUE, "reverse_window": False},
    "revdata": {"cue": ORIGINAL_CUE, "reverse_window": True},
}


def build_prompt(cue, sensor_block):
    """The shipped zero-shot template with the cue swapped in."""
    template = ex.ZERO_SHOT_TEMPLATE.replace(ORIGINAL_CUE, cue)
    if cue != ORIGINAL_CUE and ORIGINAL_CUE in ex.ZERO_SHOT_TEMPLATE and cue not in template:
        raise RuntimeError("cue substitution failed - template text drifted")
    return template.format(dataset=DATASET, n=N_CYCLES, sensor_block=sensor_block)


def sensor_block(group, reverse):
    """Same layout as experiment._sensor_block; optionally time-reversed."""
    tail = group[cfg.SENSOR_NAMES].tail(N_CYCLES)
    if reverse:
        tail = tail.iloc[::-1]
    rows = []
    for offset, (_, row) in enumerate(tail.iterrows()):
        lag = N_CYCLES - offset
        vals = "  ".join(f"{c}={v:.3f}" for c, v in row.items())
        rows.append(f"  t-{lag}: {vals}")
    return "\n".join(rows)


def run_arm(arm, test_df, test_last, client, force=False):
    spec = ARMS[arm]
    path = OUT / f"traces_p0_{DATASET}_zs_n{N_CYCLES}_{arm}.json"
    done = {}
    if path.exists() and not force:
        done = {t["engine_id"]: t for t in json.loads(path.read_text(encoding="utf-8"))}
        print(f"  [{arm}] resuming, {len(done)} engines already done")

    traces = []
    t_arm = time.time()
    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = int(row["engine_id"])
        if eid in done:
            traces.append(done[eid])
            continue

        g = test_df[test_df.engine_id == eid]
        block = sensor_block(g, spec["reverse_window"])
        prompt = build_prompt(spec["cue"], block)

        pred, reasoning = None, ""
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                raw = r.choices[0].message.content
                pred, reasoning = ex._parse_response(raw)
                break
            except Exception as e:
                print(f"    [{arm}] engine {eid} attempt {attempt + 1} failed: {e}")
                time.sleep((attempt + 1) * 5)
        if pred is None:
            print(f"    [{arm}] engine {eid} FAILED after 3 attempts - skipped")
            continue

        traces.append({"engine_id": eid, "true_rul": float(row["RUL"]),
                       "pred_rul": pred, "reasoning": reasoning, "arm": arm})
        path.write_text(json.dumps(traces, indent=2, ensure_ascii=False),
                        encoding="utf-8")

        if (i + 1) % 10 == 0:
            el = time.time() - t_arm
            print(f"    [{arm}] {i + 1}/{len(test_last)}  ({el / 60:.1f} min elapsed)",
                  flush=True)

    path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{arm}] done -> {path.name} ({len(traces)} engines, "
          f"{(time.time() - t_arm) / 60:.1f} min)")
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--engines", type=int, default=None,
                    help="limit to the first N engines (smoke test)")
    ap.add_argument("--force", action="store_true", help="ignore saved partial runs")
    args = ap.parse_args()

    cfg.N_CYCLES_IN_PROMPT = N_CYCLES
    _, test_df, test_last = ex.load_cmapss(DATASET)
    if args.engines:
        test_last = test_last.head(args.engines)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    print(f"P0.2b/P0.3 control arms - {DATASET}, zero-shot, n={N_CYCLES}, "
          f"{len(test_last)} engines, model={MODEL}, T={TEMPERATURE}")

    summary = []
    for arm in args.arms:
        print(f"\n[arm] {arm}")
        traces = run_arm(arm, test_df, test_last, client, force=args.force)
        y = np.array([t["true_rul"] for t in traces], float)
        p = np.array([t["pred_rul"] for t in traces], float)
        vals, counts = np.unique(p, return_counts=True)
        summary.append({
            "arm": arm, "n": len(p),
            "RMSE": ex.rmse(y, p), "MAE": ex.mae(y, p),
            "NASA_Score": ex.nasa_score(y, p),
            "distinct_values": int(len(vals)),
            "modal_value": float(vals[counts.argmax()]),
            "modal_share_pct": 100 * counts.max() / len(p),
            "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)),
        })
        print("   ", {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in summary[-1].items()})

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "p0_23_arm_summary.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[saved] {OUT / 'p0_23_arm_summary.csv'}")


if __name__ == "__main__":
    main()
