"""
Step 4: Semi-quantitative analysis of LLM reasoning traces.

For each sensor of physical interest, count how often the LLM:
  - mentions the sensor at all
  - identifies the correct physical direction of change with degradation
  - identifies the wrong direction (factual error / hallucination of trend)
  - mentions the sensor without specifying a direction

Outputs a markdown table suitable for direct insertion into the paper
(Section 4.5 — replacing the 3 hand-picked examples in current Table 7).

Usage:
    python analyze_explainability.py --dir results_v50_legacy --label "v50 (50 engines)"
    python analyze_explainability.py --dir results --label "v100 (100 engines)"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Ensure UTF-8 stdout on Windows consoles (default cp1252 cannot encode arrows)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Expected direction of each sensor under engine degradation
# (as asserted by the zero-shot prompt; see p1/p3_5_prompt_vs_benchmark.py)
# Column identities follow the variable list of Saxena et al. (2008): s4 is T50,
# the LPT outlet temperature, not T30, which is s3. The directions below are the
# ones the zero-shot prompt asserts; p1/p3_5_prompt_vs_benchmark.py measures what
# the benchmark actually does, and the two disagree for s9, s12 and s14.
SENSOR_PHYSICS = {
    "s4":  ("increase", "Total temperature at LPT outlet"),
    "s7":  ("decrease", "Total pressure at HPC outlet"),
    "s9":  ("decrease", "Physical core speed"),
    "s11": ("increase", "Static pressure at HPC outlet"),
    "s12": ("increase", "Ratio of fuel flow to Ps30"),
    "s14": ("decrease", "Corrected core speed"),
    "s15": ("increase", "Bypass ratio"),
}

# Patterns describing each direction (English, found in Llama outputs)
INC_WORDS = {"increas", "rising", "rises", "rise", "growing", "grows", "grew",
             "ascending", "elevated", "higher", "uptrend", "upward",
             "trending up", "trend up", "going up"}
DEC_WORDS = {"decreas", "falling", "falls", "fell", "declin", "drops", "drop",
             "descending", "lower", "downtrend", "downward",
             "trending down", "trend down", "going down", "reducing", "reduces"}
STABLE_WORDS = {"stable", "constant", "unchang", "flat", "no change", "no significant change",
                "steady", "near baseline"}


def classify_mention(text: str, sensor: str) -> str:
    """
    Look at the local context around the sensor mention and decide whether
    the model attributed an increasing, decreasing or stable trend to it.
    Returns: "increase", "decrease", "stable", or "unspecified".
    """
    # Find positions of the sensor token (case-insensitive, word-boundary)
    pattern = re.compile(rf"\b{sensor}\b", re.IGNORECASE)
    decisions = []
    for m in pattern.finditer(text):
        # local window: 60 chars before, 80 after — captures e.g. "s11 increasing (0.6→0.7)"
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 80)
        ctx = text[start:end].lower()
        # avoid spilling into the next sensor's clause
        ctx_after = text[m.end():end].lower()
        # truncate at next "; " or ". " or ", s\d" boundary
        cut = re.search(r"[;.](\s|$)|,\s*s\d", ctx_after)
        if cut:
            ctx = (text[start:m.end()] + ctx_after[:cut.start()]).lower()

        if any(w in ctx for w in INC_WORDS):
            decisions.append("increase")
        elif any(w in ctx for w in DEC_WORDS):
            decisions.append("decrease")
        elif any(w in ctx for w in STABLE_WORDS):
            decisions.append("stable")
        else:
            decisions.append("unspecified")

    if not decisions:
        return "absent"
    # If the model mentions the sensor multiple times, prefer the directional one
    for pref in ("increase", "decrease", "stable"):
        if pref in decisions:
            return pref
    return "unspecified"


def analyze_trace_file(path: Path):
    """Return dict: sensor -> Counter({correct, wrong, stable_called, unspecified, absent})."""
    with open(path) as f:
        data = json.load(f)

    stats = defaultdict(lambda: defaultdict(int))
    n_traces = len(data)
    for entry in data:
        text = entry.get("reasoning", "") or ""
        for sensor, (expected, _) in SENSOR_PHYSICS.items():
            verdict = classify_mention(text, sensor)
            if verdict == "absent":
                stats[sensor]["absent"] += 1
            elif verdict == expected:
                stats[sensor]["correct"] += 1
            elif verdict in ("increase", "decrease"):
                stats[sensor]["wrong"] += 1
            elif verdict == "stable":
                stats[sensor]["stable_called"] += 1
            else:
                stats[sensor]["unspecified"] += 1
    return stats, n_traces


def aggregate(directory: Path):
    """Aggregate stats across all trace files in a directory."""
    files = sorted(directory.glob("traces_*.json"))
    print(f"Found {len(files)} trace files in {directory}/")
    if not files:
        return None, 0

    total = defaultdict(lambda: defaultdict(int))
    grand_traces = 0
    for f in files:
        stats, n = analyze_trace_file(f)
        grand_traces += n
        for sensor, counts in stats.items():
            for k, v in counts.items():
                total[sensor][k] += v
        print(f"  {f.name:45s}  {n:4d} traces")
    return total, grand_traces


def render_markdown_table(stats, n_total, label):
    """Produce a markdown table ready for the paper."""
    lines = []
    lines.append(f"### Explainability — Semi-Quantitative Analysis ({label})")
    lines.append("")
    lines.append(f"Aggregated across **{n_total} reasoning traces** from all "
                 f"LLM configurations (zero-shot + few-shot k=3,5,10) and both sub-datasets.")
    lines.append("")
    lines.append("| Sensor | Physical Quantity | Expected Trend | Correctly Identified | Wrong Direction | Mentioned but Unspecified | Not Mentioned |")
    lines.append("|---|---|---|---|---|---|---|")
    for sensor, (expected, descr) in SENSOR_PHYSICS.items():
        c = stats[sensor]
        correct = c["correct"]
        wrong = c["wrong"] + c.get("stable_called", 0)  # stable when expected drift = wrong
        unspec = c.get("unspecified", 0)
        absent = c.get("absent", 0)
        total = correct + wrong + unspec + absent
        if total == 0:
            continue
        arrow = "↑" if expected == "increase" else ("↓" if expected == "decrease" else "—")
        lines.append(f"| {sensor} | {descr} | {arrow} ({expected}) | "
                     f"{correct} ({correct/total*100:.0f}%) | "
                     f"{wrong} ({wrong/total*100:.0f}%) | "
                     f"{unspec} ({unspec/total*100:.0f}%) | "
                     f"{absent} ({absent/total*100:.0f}%) |")
    lines.append("")
    # Summary statistics across the 7 physically informative sensors
    n_sensors = len(SENSOR_PHYSICS)
    tot_correct = sum(stats[s]["correct"] for s in SENSOR_PHYSICS)
    # A "directional claim" is one that names a direction: increase or decrease.
    # Verdicts of "stable" are counted on their own line. Release 1.0 of this
    # script folded them into the wrong-direction count and therefore into this
    # denominator, which is why its report read 3,825 claims and 91.1% where
    # this one reads 3,816 and 91.3% on the same traces.
    tot_wrong = sum(stats[s]["wrong"] for s in SENSOR_PHYSICS)
    tot_stable = sum(stats[s].get("stable_called", 0) for s in SENSOR_PHYSICS)
    tot_dir = tot_correct + tot_wrong
    lines.append("**Summary across the 7 informative sensors:**")
    lines.append(f"- Directional claims (increase or decrease): **{tot_dir}** "
                 f"({tot_dir / (n_total * n_sensors) * 100:.1f}% "
                 f"of {n_total * n_sensors} sensor-trace opportunities).")
    if tot_dir > 0:
        lines.append(f"- Of these, **{tot_correct}** ({tot_correct / tot_dir * 100:.1f}%) name the "
                     f"canonical degradation direction and **{tot_wrong}** "
                     f"({tot_wrong / tot_dir * 100:.1f}%) name the opposite one.")
    lines.append(f"- A further **{tot_stable}** mentions call the sensor stable; they state no "
                 "direction and are excluded from the two percentages above.")
    lines.append("")
    lines.append("Agreement with the canonical direction is not a measure of whether the "
                 "explanation describes the input. For that, see "
                 "`p0/p0_2a_faithfulness.py` and `results_p0/p0_2a_report.md`, which score the "
                 "same claims against the sensor window actually shown in each prompt.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="results_v50_legacy",
                   help="Directory containing traces_*.json files")
    p.add_argument("--label", type=str, default="50-engine subset",
                   help="Label for the report (e.g. 'v100 — full test set')")
    p.add_argument("--out", type=str, default=None,
                   help="Optional output markdown file")
    args = p.parse_args()

    directory = Path(args.dir)
    if not directory.exists():
        print(f"ERROR: directory '{directory}' not found")
        return

    stats, n_total = aggregate(directory)
    if stats is None:
        return

    md = render_markdown_table(stats, n_total, args.label)
    print("\n" + "=" * 65)
    print(md)
    print("=" * 65)

    out = Path(args.out) if args.out else directory / "explainability_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()
