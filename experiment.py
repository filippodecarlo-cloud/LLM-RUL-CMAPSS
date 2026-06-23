"""
Main experiment: RUL prediction on NASA CMAPSS
- Baselines: Random Forest, LSTM
- LLM: Llama 3.1 8B via Ollama (zero-shot, few-shot k=3,5,10)
- Metrics: RMSE, MAE, NASA Scoring Function

CLI usage examples:
  python experiment.py                                    # default config
  python experiment.py --cycles 30 --suffix n30           # ablation 30-cycle window
  python experiment.py --datasets FD001 --skip-baselines  # LLM only, FD001 only
  python experiment.py --cycles 30 --suffix n30_run2 --datasets FD001 --modes zero_shot --temperature 0.1
                                                          # repeat run for variance estimate
"""

import argparse
import time
import json
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from openai import OpenAI

import config as cfg
from config import (DATASETS, RUL_CAP, FEW_SHOT_K, API_DELAY,
                    SENSOR_NAMES, MAX_TEST_ENGINES)

warnings.filterwarnings("ignore")

DATA_DIR  = Path("cmapss_data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ALL_COLS = (["engine_id", "cycle", "setting1", "setting2", "setting3"] +
            [f"s{i}" for i in range(1, 22)])
CONTEXT_WINDOW = 30  # cycles fed to LSTM/RF window features


# ────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ────────────────────────────────────────────────────────────

def load_cmapss(dataset: str):
    """Return (train_df, test_df, test_last_df) for one sub-dataset."""
    def _read(path):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS,
                         engine="python").dropna(axis=1)
        return df

    train = _read(DATA_DIR / f"train_{dataset}.txt")
    test  = _read(DATA_DIR / f"test_{dataset}.txt")
    rul   = pd.read_csv(DATA_DIR / f"RUL_{dataset}.txt",
                        sep=r"\s+", header=None, names=["RUL"],
                        engine="python").dropna(axis=1)

    # Piecewise-linear RUL for training
    maxc = train.groupby("engine_id")["cycle"].max().rename("max_cycle")
    train = train.join(maxc, on="engine_id")
    train["RUL"] = (train["max_cycle"] - train["cycle"]).clip(upper=RUL_CAP)
    train.drop(columns="max_cycle", inplace=True)

    # RUL for test (last observed cycle of each engine)
    test_last = test.groupby("engine_id").last().reset_index()
    test_last["RUL"] = rul["RUL"].clip(upper=RUL_CAP).values

    # Normalise sensors
    scaler = MinMaxScaler()
    train[SENSOR_NAMES]    = scaler.fit_transform(train[SENSOR_NAMES])
    test[SENSOR_NAMES]     = scaler.transform(test[SENSOR_NAMES])
    test_last[SENSOR_NAMES] = scaler.transform(test_last[SENSOR_NAMES])

    return train, test, test_last


# ────────────────────────────────────────────────────────────
# 2. METRICS
# ────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())

def evaluate(y_true, y_pred, label: str) -> dict:
    r, m, s = rmse(y_true, y_pred), mae(y_true, y_pred), nasa_score(y_true, y_pred)
    print(f"  {label:45s}  RMSE={r:7.2f}  MAE={m:7.2f}  NASA={s:10.1f}")
    return {"label": label, "RMSE": r, "MAE": m, "NASA_Score": s}


# ────────────────────────────────────────────────────────────
# 3. RANDOM FOREST BASELINE
# ────────────────────────────────────────────────────────────

def _window_features(group: pd.DataFrame) -> np.ndarray:
    w = group[SENSOR_NAMES].values
    if len(w) >= CONTEXT_WINDOW:
        w = w[-CONTEXT_WINDOW:]
    else:
        w = np.vstack([np.zeros((CONTEXT_WINDOW - len(w), len(SENSOR_NAMES))), w])
    return np.concatenate([w.mean(0), w.std(0), w[-1] - w[0]])


def run_rf(train_df, test_df, test_last):
    # Use ALL sliding windows per engine (not just the last one)
    X_tr, y_tr = [], []
    for _, g in train_df.groupby("engine_id"):
        vals = g[SENSOR_NAMES].values
        ruls = g["RUL"].values
        for i in range(CONTEXT_WINDOW, len(vals) + 1):
            w = vals[i - CONTEXT_WINDOW:i]
            X_tr.append(np.concatenate([w.mean(0), w.std(0), w[-1] - w[0]]))
            y_tr.append(ruls[i - 1])
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    preds = []
    for _, row in test_last.iterrows():
        g = test_df[test_df["engine_id"] == row["engine_id"]]
        preds.append(rf.predict([_window_features(g)])[0])

    return np.clip(preds, 0, RUL_CAP), test_last["RUL"].values


# ────────────────────────────────────────────────────────────
# 4. LSTM BASELINE
# ────────────────────────────────────────────────────────────

def run_lstm(train_df, test_df, test_last):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError:
        print("  [SKIP] TensorFlow not found — run: pip install tensorflow")
        return None, None

    X, y = [], []
    for _, g in train_df.groupby("engine_id"):
        vals = g[SENSOR_NAMES].values
        for i in range(len(vals) - CONTEXT_WINDOW + 1):
            X.append(vals[i:i + CONTEXT_WINDOW])
            y.append(g["RUL"].iloc[i + CONTEXT_WINDOW - 1])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, input_shape=(CONTEXT_WINDOW, len(SENSOR_NAMES)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, batch_size=256, validation_split=0.1, verbose=0)

    preds = []
    for _, row in test_last.iterrows():
        g = test_df[test_df["engine_id"] == row["engine_id"]]
        w = g[SENSOR_NAMES].values
        if len(w) < CONTEXT_WINDOW:
            w = np.vstack([np.zeros((CONTEXT_WINDOW - len(w), len(SENSOR_NAMES))), w])
        else:
            w = w[-CONTEXT_WINDOW:]
        p = model.predict(w.reshape(1, CONTEXT_WINDOW, len(SENSOR_NAMES)), verbose=0)[0][0]
        preds.append(p)

    return np.clip(preds, 0, RUL_CAP), test_last["RUL"].values


# ────────────────────────────────────────────────────────────
# 5. LLM PROMPTING
# ────────────────────────────────────────────────────────────

def _sensor_block(group: pd.DataFrame) -> str:
    n = cfg.N_CYCLES_IN_PROMPT
    tail = group[SENSOR_NAMES].tail(n)
    rows = []
    for offset, (_, row) in enumerate(tail.iterrows()):
        lag = n - offset
        vals = "  ".join(f"{c}={v:.3f}" for c, v in row.items())
        rows.append(f"  t-{lag}: {vals}")
    return "\n".join(rows)


ZERO_SHOT_TEMPLATE = """\
You are an expert predictive maintenance engineer.

TASK: Estimate the Remaining Useful Life (RUL) of a turbofan engine.

BACKGROUND:
- Dataset: NASA CMAPSS {dataset}
- Sensor readings are min-max normalised (0 = minimum observed, 1 = maximum observed)
- RUL = number of operational cycles remaining before failure
- Valid RUL range: 0 (imminent failure) to 125 (healthy / far from failure)

RECENT SENSOR READINGS (last {n} cycles, oldest first):
{sensor_block}

INSTRUCTIONS:
Analyse the sensor trends. Higher values for degradation-related sensors (e.g., s11, s12, s15)
and lower values for efficiency-related sensors (e.g., s9, s14) typically indicate advanced wear.
Increasing variance or monotonic drift also suggests approaching failure.

Reply in EXACTLY this format — no extra text:
RUL_ESTIMATE: <integer 0-125>
REASONING: <2-3 sentences explaining the dominant sensor trends and degradation pattern>
"""

FEW_SHOT_TEMPLATE = """\
You are an expert predictive maintenance engineer.

TASK: Estimate the Remaining Useful Life (RUL) of a turbofan engine.

BACKGROUND:
- Dataset: NASA CMAPSS {dataset}
- Sensor readings are min-max normalised (0–1). RUL range: 0–125.

REFERENCE EXAMPLES from similar engines:
{examples}

NOW EVALUATE THIS ENGINE:
SENSOR READINGS (last {n} cycles, oldest first):
{sensor_block}

Reply in EXACTLY this format:
RUL_ESTIMATE: <integer 0-125>
REASONING: <2-3 sentences>
"""


def _parse_response(text: str) -> tuple[int, str]:
    rul, reasoning = None, ""
    for line in text.splitlines():
        if line.startswith("RUL_ESTIMATE:"):
            m = re.search(r"\d+", line)
            if m:
                rul = int(np.clip(int(m.group()), 0, RUL_CAP))
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    if rul is None:
        nums = re.findall(r"\b(\d{1,3})\b", text)
        rul = int(np.clip(int(nums[0]), 0, RUL_CAP)) if nums else 75
    return rul, reasoning


def _build_example_block(examples: list[dict]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        rul_label = ex["true_rul"]
        if rul_label > 80:
            desc = "stable operation; sensors near baseline"
        elif rul_label > 40:
            desc = "progressive degradation; moderate drift in key sensors"
        else:
            desc = "advanced wear; significant deviation in efficiency sensors"
        lines.append(
            f"Example {i} — True RUL: {rul_label}\n"
            f"Sensor readings:\n{ex['sensor_block']}\n"
            f"Pattern: {desc}\n"
        )
    return "\n".join(lines)


def run_llm(train_df, test_df, test_last, dataset: str,
            mode: str = "zero_shot", k: int = 0,
            suffix: str = "", temperature: float = 0.1) -> tuple[np.ndarray, np.ndarray, list]:

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model_name = "llama3.1:8b"

    # Build few-shot pool (stratified by RUL range)
    examples = []
    if mode == "few_shot" and k > 0:
        pool = train_df.groupby("engine_id").last().reset_index()
        pool["bin"] = pd.cut(pool["RUL"], bins=[0, 40, 80, 125],
                             labels=[0, 1, 2], include_lowest=True).astype(int)
        per_bin = max(1, k // 3)
        chosen_ids = []
        for b in range(3):
            ids = pool[pool["bin"] == b]["engine_id"].tolist()
            chosen_ids.extend(ids[:per_bin])
        chosen_ids = chosen_ids[:k]

        for eid in chosen_ids:
            g = train_df[train_df["engine_id"] == eid]
            examples.append({
                "sensor_block": _sensor_block(g),
                "true_rul": int(g["RUL"].iloc[-1])
            })

    # Limit engines if MAX_TEST_ENGINES is set in config
    if MAX_TEST_ENGINES:
        test_last = test_last.head(MAX_TEST_ENGINES)

    preds, true_ruls, traces = [], [], []
    total = len(test_last)
    print(f"\n  Running LLM [{mode}, k={k}, n_cycles={cfg.N_CYCLES_IN_PROMPT}, T={temperature}] "
          f"on {dataset} ({total} engines)...")

    for i, (_, row) in enumerate(test_last.iterrows()):
        eid = row["engine_id"]
        g = test_df[test_df["engine_id"] == eid]
        block = _sensor_block(g)

        if mode == "zero_shot":
            prompt = ZERO_SHOT_TEMPLATE.format(
                dataset=dataset, n=cfg.N_CYCLES_IN_PROMPT, sensor_block=block)
        else:
            prompt = FEW_SHOT_TEMPLATE.format(
                dataset=dataset, n=cfg.N_CYCLES_IN_PROMPT,
                examples=_build_example_block(examples),
                sensor_block=block)

        pred_rul, reasoning = 75, "API error"
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                pred_rul, reasoning = _parse_response(resp.choices[0].message.content)
                break
            except Exception as e:
                wait = (attempt + 1) * 10
                print(f"    Attempt {attempt+1} failed (engine {eid}): retrying in {wait}s...")
                time.sleep(wait)

        preds.append(pred_rul)
        true_ruls.append(float(row["RUL"]))
        traces.append({
            "engine_id": int(eid),
            "true_rul": float(row["RUL"]),
            "pred_rul": pred_rul,
            "reasoning": reasoning
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{total} completed")

        time.sleep(0.5)  # small pause between local calls

    # Save reasoning traces for qualitative analysis
    suf = f"_{suffix}" if suffix else ""
    tag = f"{dataset}_{mode}_k{k}{suf}"
    trace_path = str((RESULTS_DIR / f"traces_{tag}.json").resolve())
    for _attempt in range(3):
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(traces, f, indent=2, ensure_ascii=False)
            break
        except OSError as e:
            print(f"  WARNING: trace save attempt failed ({e}); retrying…")
            time.sleep(2)
    else:
        print(f"  ERROR: could not save traces to {trace_path}")

    return np.clip(preds, 0, RUL_CAP), np.array(true_ruls)


# ────────────────────────────────────────────────────────────
# 6. MAIN
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CMAPSS RUL prediction experiment")
    p.add_argument("--cycles", type=int, default=cfg.N_CYCLES_IN_PROMPT,
                   help=f"Cycles in LLM prompt (default {cfg.N_CYCLES_IN_PROMPT}; ablation: 5/15/30)")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   help=f"Sub-datasets (default: {DATASETS})")
    p.add_argument("--modes", nargs="+", default=["zero_shot", "few_shot"],
                   choices=["zero_shot", "few_shot"],
                   help="Which LLM modes to run")
    p.add_argument("--ks", nargs="+", type=int, default=FEW_SHOT_K,
                   help=f"Few-shot k values (default {FEW_SHOT_K})")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM sampling temperature (default 0.1)")
    p.add_argument("--suffix", type=str, default="",
                   help="Suffix appended to trace filenames (e.g. 'n30', 'run2')")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip Random Forest and LSTM (LLM only)")
    p.add_argument("--engines", type=int, default=None,
                   help="Override MAX_TEST_ENGINES (default: from config)")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to config module (read by run_llm via cfg.N_CYCLES_IN_PROMPT)
    cfg.N_CYCLES_IN_PROMPT = args.cycles
    if args.engines is not None:
        global MAX_TEST_ENGINES
        MAX_TEST_ENGINES = args.engines

    print("=" * 65)
    print("RUL Prediction Experiment — CMAPSS + Llama 3.1 8B (Ollama)")
    print("=" * 65)
    print(f"Config: cycles={cfg.N_CYCLES_IN_PROMPT} | datasets={args.datasets} | "
          f"modes={args.modes} | ks={args.ks} | T={args.temperature} | "
          f"engines={MAX_TEST_ENGINES if MAX_TEST_ENGINES else 'ALL'} | suffix='{args.suffix}'")
    print(f"Skip baselines: {args.skip_baselines}")

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'─' * 65}")
        print(f"Dataset: {dataset}")
        print(f"{'─' * 65}")

        try:
            train_df, test_df, test_last = load_cmapss(dataset)
        except FileNotFoundError:
            print(f"  [SKIP] Files for {dataset} not found in {DATA_DIR}/")
            continue

        y_true = test_last["RUL"].values
        if MAX_TEST_ENGINES:
            y_true = y_true[:MAX_TEST_ENGINES]

        if not args.skip_baselines:
            print("\n[Baseline] Random Forest")
            rf_preds, rf_true = run_rf(train_df, test_df,
                                       test_last.head(MAX_TEST_ENGINES) if MAX_TEST_ENGINES else test_last)
            r = evaluate(rf_true, rf_preds, f"RandomForest | {dataset}")
            r["dataset"], r["model"] = dataset, "RandomForest"
            all_results.append(r)

            print("\n[Baseline] LSTM")
            lstm_preds, lstm_true = run_lstm(train_df, test_df,
                                             test_last.head(MAX_TEST_ENGINES) if MAX_TEST_ENGINES else test_last)
            if lstm_preds is not None:
                r = evaluate(lstm_true, lstm_preds, f"LSTM | {dataset}")
                r["dataset"], r["model"] = dataset, "LSTM"
                all_results.append(r)

        if "zero_shot" in args.modes:
            print("\n[LLM] Llama 3.1 8B — Zero-Shot")
            zs_preds, zs_true = run_llm(train_df, test_df, test_last,
                                         dataset, mode="zero_shot",
                                         suffix=args.suffix, temperature=args.temperature)
            r = evaluate(zs_true, zs_preds, f"Llama-ZeroShot | {dataset}")
            r["dataset"], r["model"], r["k"] = dataset, "Llama-ZeroShot", 0
            r["n_cycles"], r["temperature"], r["suffix"] = args.cycles, args.temperature, args.suffix
            all_results.append(r)

        if "few_shot" in args.modes:
            for k in args.ks:
                print(f"\n[LLM] Llama 3.1 8B — Few-Shot k={k}")
                fs_preds, fs_true = run_llm(train_df, test_df, test_last,
                                             dataset, mode="few_shot", k=k,
                                             suffix=args.suffix, temperature=args.temperature)
                r = evaluate(fs_true, fs_preds, f"Llama-FewShot-k{k} | {dataset}")
                r["dataset"], r["model"], r["k"] = dataset, f"Llama-FewShot-k{k}", k
                r["n_cycles"], r["temperature"], r["suffix"] = args.cycles, args.temperature, args.suffix
                all_results.append(r)

    results_df = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / (f"all_results_{args.suffix}.csv" if args.suffix else "all_results.csv")
    results_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(results_df[["dataset", "model", "RMSE", "MAE", "NASA_Score"]].to_string(index=False))
    print(f"\nAll results saved to:  {out_csv}")
    print(f"Reasoning traces in:   {RESULTS_DIR.absolute()}/traces_*.json")


if __name__ == "__main__":
    main()
