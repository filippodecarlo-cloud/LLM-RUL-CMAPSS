"""
Build a machine-readable manifest of every inference run in the repository.

One record per trace file: the configuration it came from, how many responses it
holds, its SHA-256, and the summary statistics recomputed from the file itself
rather than copied from a report. A reviewer can check any number in the paper
against this without reading Python, and can check the files have not changed.

    python p1/make_manifest.py [--repo <path>]

Writes <repo>/manifest.json and <repo>/manifest.csv.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

RUL_CAP = 125


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nasa_score(err):
    """Positive error = late prediction, which the score penalises harder."""
    err = np.asarray(err, dtype=float)
    return float(np.sum(np.where(err < 0,
                                 np.exp(-err / 13.0) - 1.0,
                                 np.exp(err / 10.0) - 1.0)))


def describe(path):
    """Everything derivable from the trace file alone."""
    traces = json.loads(path.read_text(encoding="utf-8"))
    pred = np.array([t["pred_rul"] for t in traces], dtype=float)
    true = np.array([t["true_rul"] for t in traces], dtype=float)
    err = pred - true
    counts = Counter(pred.tolist())
    n = len(pred)
    p = np.array([c / n for c in counts.values()])
    modal_value, modal_n = counts.most_common(1)[0]
    reasoning = [t.get("reasoning", "") or "" for t in traces]
    return {
        "engines": n,
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "nasa_score": round(nasa_score(err), 1),
        "distinct_values": len(counts),
        "modal_value": float(modal_value),
        "modal_share_pct": round(100.0 * modal_n / n, 1),
        "entropy_bits": round(float(-np.sum(p * np.log2(p))), 4),
        "pred_mean": round(float(pred.mean()), 3),
        "true_mean": round(float(true.mean()), 3),
        "empty_reasoning": sum(1 for r in reasoning if not r.strip()),
        "reasoning_chars_mean": round(float(np.mean([len(r) for r in reasoning])), 1),
    }


# Which experiment each family of trace files belongs to. The distinction matters:
# claims true of the original grid are not automatically true of the others.
def classify(rel):
    name = Path(rel).name
    if rel.startswith("results/"):
        return "original grid", "27 runs on FD001 and FD003, k x n, original example sampler"
    if rel.startswith("results_p0/"):
        return "control arms", "paired prompt controls on FD001 zero-shot n=30"
    if "_grid_" in name:
        return "corrected grid", "k x n re-run with the stratified example sampler, FD001"
    if re.search(r"_fs_k5_n5_(asis|strat|random)_seed", name):
        return "example-set experiment", "k fixed, only the examples change, FD001"
    if name.startswith("traces_p2_"):
        return "multi-condition", "FD002 and FD004, zero-shot n=30, per-regime scaling"
    return "model sweep", "one configuration, different models, FD001 zero-shot n=30"


def parse_config(rel):
    """Best-effort configuration fields, from the filename convention."""
    name = Path(rel).stem
    out = {}
    m = re.search(r"FD00\d", name)
    if m:
        out["dataset"] = m.group(0)
    if "zero_shot" in name or "_zs_" in name:
        out["mode"], out["k"] = "zero_shot", 0
    elif "few_shot" in name or "_fs_" in name:
        out["mode"] = "few_shot"
    m = re.search(r"(?:_k|_fs_k)(\d+)", name)
    if m and out.get("mode") != "zero_shot":
        out["k"] = int(m.group(1))
    m = re.search(r"_n(\d+)", name)
    out["n_cycles"] = int(m.group(1)) if m else 5
    m = re.search(r"_seed(\d+)", name)
    if m:
        out["example_seed"] = int(m.group(1))
    m = re.search(r"_(run\d)", name)
    out["replicate"] = m.group(1) if m else "run1"
    # model: named in the filename only where it is not the reference model
    for tag, pretty in [("deepseek-r1_7b", "deepseek-r1:7b"),
                        ("qwen2.5_7b", "qwen2.5:7b"),
                        ("mistral_7b", "mistral:7b"),
                        ("llama3.1_8b-instruct-q8_0", "llama3.1:8b-instruct-q8_0")]:
        if tag in name:
            out["model"] = pretty
            break
    else:
        out["model"] = "llama3.1:8b"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(EXP.parent / "LLM-RUL-CMAPSS"))
    args = ap.parse_args()
    repo = Path(args.repo)

    records = []
    for folder in ("results", "results_p0", "results_p1"):
        for path in sorted((repo / folder).glob("traces_*.json")):
            rel = f"{folder}/{path.name}"
            experiment, note = classify(rel)
            records.append({
                "file": rel,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "experiment": experiment,
                "experiment_note": note,
                **parse_config(rel),
                **describe(path),
            })

    total = sum(r["engines"] for r in records)
    by_exp = Counter(r["experiment"] for r in records)
    manifest = {
        "generated": date.today().isoformat(),
        "generator": "p1/make_manifest.py",
        "total_trace_files": len(records),
        "total_responses": total,
        "responses_by_experiment": {
            k: sum(r["engines"] for r in records if r["experiment"] == k)
            for k in sorted(by_exp)},
        "notes": [
            "Metrics are recomputed from each trace file, not copied from a report.",
            "NASA score uses exp(-e/13)-1 for early predictions and exp(e/10)-1 for "
            "late ones, where e = predicted minus true.",
            "No sampling seed was sent to the model, so re-running inference will not "
            "reproduce these files; see REPRODUCIBILITY.md.",
        ],
        "runs": records,
    }
    (repo / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    cols = list(dict.fromkeys(k for r in records for k in r))
    with open(repo / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"{len(records)} trace files, {total:,} responses")
    for k in sorted(by_exp):
        n = sum(r["engines"] for r in records if r["experiment"] == k)
        print(f"  {k:26s} {by_exp[k]:3d} files  {n:5,d} responses")
    print(f"[saved] {repo / 'manifest.json'}")
    print(f"[saved] {repo / 'manifest.csv'}")


if __name__ == "__main__":
    main()
