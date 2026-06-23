"""
Verifies Table 8 of the paper by computing distinct predicted values and
top-1 share from all 12 FD001 trace files.

Also prints the (true_rul, pred_rul) pair for engine 1 in the zero-shot n=5
trace, needed to confirm Appendix A.5.

Run from Anaconda Prompt:
    cd "K:\\Il mio Drive\\02 UNI\\Articoli\\2026 AI SI\\experiment"
    python check_table8.py
"""

import json
from collections import Counter
from pathlib import Path

RESULTS = Path("results")

CONFIGS = [
    ("Zero-Shot",     5,  "traces_FD001_zero_shot_k0.json"),
    ("Zero-Shot",     15, "traces_FD001_zero_shot_k0_n15.json"),
    ("Zero-Shot",     30, "traces_FD001_zero_shot_k0_n30.json"),
    ("Few-Shot k=3",  5,  "traces_FD001_few_shot_k3.json"),
    ("Few-Shot k=3",  15, "traces_FD001_few_shot_k3_n15.json"),
    ("Few-Shot k=3",  30, "traces_FD001_few_shot_k3_n30.json"),
    ("Few-Shot k=5",  5,  "traces_FD001_few_shot_k5.json"),
    ("Few-Shot k=5",  15, "traces_FD001_few_shot_k5_n15.json"),
    ("Few-Shot k=5",  30, "traces_FD001_few_shot_k5_n30.json"),
    ("Few-Shot k=10", 5,  "traces_FD001_few_shot_k10.json"),
    ("Few-Shot k=10", 15, "traces_FD001_few_shot_k10_n15.json"),
    ("Few-Shot k=10", 30, "traces_FD001_few_shot_k10_n30.json"),
]


def load_traces(fname):
    with open(RESULTS / fname, encoding="utf-8") as f:
        return json.load(f)


def summarise(data):
    preds = [d["pred_rul"] for d in data]
    c = Counter(preds)
    n_distinct = len(c)
    top1_value, top1_count = c.most_common(1)[0]
    top1_share = 100 * top1_count / len(preds)
    full_dist = c.most_common()  # all (value, count) pairs sorted
    return n_distinct, top1_value, top1_share, full_dist


# ── Table 8 ──────────────────────────────────────────────────────
print("=" * 70)
print("TABLE 8 (real) — FD001 predicted-value distribution")
print("=" * 70)
print(f"{'Config':<14}{'n':>4}{'Distinct':>10}{'Top-1':>10}{'Share':>10}")
print("-" * 70)
for label, n, fname in CONFIGS:
    try:
        data = load_traces(fname)
        nd, t1v, t1s, _ = summarise(data)
        print(f"{label:<14}{n:>4}{nd:>10}{t1v:>10}{t1s:>9.1f}%")
    except FileNotFoundError:
        print(f"{label:<14}{n:>4}  MISSING ({fname})")

# ── Full distribution per config ─────────────────────────────────
print("\n" + "=" * 70)
print("Full predicted-value distribution per configuration")
print("=" * 70)
for label, n, fname in CONFIGS:
    try:
        data = load_traces(fname)
        _, _, _, dist = summarise(data)
        dist_str = ", ".join(f"{v}:{c}" for v, c in dist)
        print(f"{label} n={n}: {dist_str}")
    except FileNotFoundError:
        pass

# ── Engine 1 in ZS n=5 (for Appendix A.5) ────────────────────────
print("\n" + "=" * 70)
print("Engine 1 in zero-shot n=5 (Appendix A.5)")
print("=" * 70)
try:
    data = load_traces("traces_FD001_zero_shot_k0.json")
    # try by engine_id field, then fall back to first record
    engine1 = next(
        (d for d in data if d.get("engine_id") == 1 or d.get("unit_id") == 1),
        data[0],
    )
    print(f"keys available: {list(engine1.keys())}")
    print(f"true_rul = {engine1.get('true_rul')}")
    print(f"pred_rul = {engine1.get('pred_rul')}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
