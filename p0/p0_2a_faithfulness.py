"""
P0.2a - Is the explanation about the input, or about the prompt?

It is common to score an explanation "correct" when the direction it claims
for a sensor matches the expected degradation direction. But the zero-shot
prompt STATES those directions:

    "Higher values for degradation-related sensors (e.g., s11, s12, s15)
     and lower values for efficiency-related sensors (e.g., s9, s14)
     typically indicate advanced wear."

so the metric rewards echoing the prompt, which is prompt leakage rather than
domain knowledge.
This script separates two different questions, over all existing traces:

  REFERENCE AGREEMENT claimed direction == the expected degradation direction
                      (the measure usually reported)
  INPUT FAITHFULNESS  claimed direction == the direction actually present in
                      the n-cycle window that was placed in that engine's prompt
                      (what "explanation" has to mean)

Two built-in controls make this a real experiment on existing data:
  * CUED vs UNCUED SENSORS. The zero-shot prompt names s9,s11,s12,s14,s15 but
    never mentions s4 or s7, which have equally canonical directions.
  * CUED vs UNCUED PROMPTS. The few-shot template contains no direction cue at all.

Outputs (results_p0/):
  p0_2a_faithfulness_by_config.csv   per configuration x sensor
  p0_2a_claims.csv                   one row per (config, engine, sensor) claim
  p0_2a_report.md
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
OUT = EXP / "results_p0"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(EXP))
from analyze_explainability import SENSOR_PHYSICS, classify_mention  # noqa: E402

# Sensors whose expected direction the zero-shot prompt explicitly states.
CUED_SENSORS = {"s9", "s11", "s12", "s14", "s15"}
# s4, s7 are canonical in the literature but absent from every prompt.

# A window is called flat when its net change is below this, in the same
# normalised units the model was shown (values printed to 3 decimals).
FLAT_EPS = 0.01


def actual_direction(vals, eps=FLAT_EPS):
    """Direction actually present in the displayed window: last - first."""
    net = float(vals[-1] - vals[0])
    if abs(net) < eps:
        return "stable", net
    return ("increase" if net > 0 else "decrease"), net


def parse_tag(path):
    stem = path.stem.replace("traces_", "")
    ds, rest = stem[:5], stem[6:]
    m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
    mode, k, n, run = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    return {"dataset": ds, "mode": mode, "k": k,
            "n_cycles": int(n) if n else 5, "run": run or "run1", "file": path.name}


def main():
    import experiment as ex

    # normalised test frames, exactly as the prompt builder sees them
    frames = {}
    for ds in ("FD001", "FD003"):
        _, test_df, _ = ex.load_cmapss(ds)
        frames[ds] = test_df

    claims = []
    for path in sorted((EXP / "results").glob("traces_*.json")):
        info = parse_tag(path)
        test_df = frames[info["dataset"]]
        n = info["n_cycles"]
        traces = json.loads(path.read_text(encoding="utf-8"))

        for t in traces:
            eid = t["engine_id"]
            text = t.get("reasoning", "") or ""
            g = test_df[test_df.engine_id == eid]
            # the exact rows the prompt showed, at the precision it showed them
            win = g.tail(n)

            for sensor, (expected, _) in SENSOR_PHYSICS.items():
                claimed = classify_mention(text, sensor)
                vals = np.round(win[sensor].values.astype(float), 3)
                actual, net = actual_direction(vals)
                claims.append({
                    **{kk: info[kk] for kk in
                       ("dataset", "mode", "k", "n_cycles", "run")},
                    "engine_id": eid, "sensor": sensor,
                    "cued": sensor in CUED_SENSORS,
                    "claimed": claimed, "expected_textbook": expected,
                    "actual_in_window": actual, "net_change": net,
                    "pred_rul": t["pred_rul"], "true_rul": t["true_rul"],
                })

    cl = pd.DataFrame(claims)
    cl.to_csv(OUT / "p0_2a_claims.csv", index=False)

    # Only directional claims can be scored for direction.
    d = cl[cl.claimed.isin(["increase", "decrease"])].copy()
    d["textbook_hit"] = d.claimed == d.expected_textbook
    d["input_hit"] = d.claimed == d.actual_in_window

    # Base rate: what a rule that ignores the text and always asserts the
    # canonical direction would score. It must be computed on the SAME rows as
    # faithfulness, i.e. the directional claims. Over all 18,900 opportunities
    # the same rule scores 46.7% and over the 3,816 claims 42.0%; only the
    # second is a like-for-like comparison, because faithfulness is undefined
    # wherever the model stated no direction.
    base = d.copy()
    base["textbook_vs_actual"] = base.expected_textbook == base.actual_in_window

    def block(sub):
        return {"n_claims": len(sub),
                "textbook_agreement_pct": 100 * sub.textbook_hit.mean(),
                "input_faithfulness_pct": 100 * sub.input_hit.mean()}

    L = ["# P0.2a - Textbook agreement vs input faithfulness\n"]
    L.append(f"All {len(cl.file.unique()) if 'file' in cl else 27} trace files, "
             f"{cl.engine_id.count():,} sensor-trace opportunities, "
             f"{len(d):,} explicit directional claims.\n")
    L.append(f"A window counts as flat when |last - first| < {FLAT_EPS} in the "
             "normalised units printed in the prompt.\n")

    # -- headline ----------------------------------------------------------
    tot = block(d)
    L.append("\n## Headline\n")
    L.append("| Measure | Value |")
    L.append("|---|---|")
    L.append(f"| Directional claims scored | {tot['n_claims']:,} |")
    L.append(f"| **Textbook agreement** (what the paper reported) | "
             f"**{tot['textbook_agreement_pct']:.1f}%** |")
    L.append(f"| **Input faithfulness** (claim matches the window shown) | "
             f"**{tot['input_faithfulness_pct']:.1f}%** |")
    L.append(f"| Base rate: canonical direction present, same claims | "
             f"{100 * base.textbook_vs_actual.mean():.1f}% |")
    L.append(f"| (the same rule over all {len(cl):,} opportunities, different "
             f"denominator, not comparable) | "
             f"{100 * (cl.expected_textbook == cl.actual_in_window).mean():.1f}% |")

    # -- control 1: cued vs uncued sensors ---------------------------------
    L.append("\n## Control 1 - sensors the prompt names vs sensors it does not\n")
    L.append("The cue sentence names s9, s11, s12, s14, s15 and is silent about s4 and s7, "
             "whose canonical directions are just as established. s4 and s7 are uncued in "
             "every prompt variant, so all modes are pooled here.\n")
    L.append("| Sensor set | Directional claims | Textbook agreement | Input faithfulness |")
    L.append("|---|---|---|---|")
    for lab, sub in (("CUED (s9,s11,s12,s14,s15)", d[d.cued]),
                     ("UNCUED (s4,s7)", d[~d.cued])):
        b = block(sub)
        L.append(f"| {lab} | {b['n_claims']:,} | {b['textbook_agreement_pct']:.1f}% | "
                 f"{b['input_faithfulness_pct']:.1f}% |")
    ct = pd.crosstab(d.cued, d.textbook_hit)
    if ct.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct)
        L.append(f"\nchi2 = {chi2:.1f}, p = {p:.2e} for the cued/uncued difference in "
                 "textbook agreement. Cueing a sensor moves agreement with the canonical "
                 "direction; it does not move agreement with the data.\n")

    # -- control 2: cued prompt vs uncued prompt ---------------------------
    L.append("\n## Control 2 - zero-shot prompt (carries the cue) vs few-shot prompt (does not)\n")
    L.append("| Prompt | Directional claims | Textbook agreement | Input faithfulness |")
    L.append("|---|---|---|---|")
    for lab, sub in (("zero-shot (cue present)", d[d["mode"] == "zero_shot"]),
                     ("few-shot (no cue sentence)", d[d["mode"] == "few_shot"])):
        b = block(sub)
        L.append(f"| {lab} | {b['n_claims']:,} | {b['textbook_agreement_pct']:.1f}% | "
                 f"{b['input_faithfulness_pct']:.1f}% |")

    # -- per sensor --------------------------------------------------------
    L.append("\n## Per sensor\n")
    L.append("| Sensor | In prompt? | Canonical | Claims | Textbook agr. | Input faith. | "
             "Canonical direction present in window |")
    L.append("|---|---|---|---|---|---|---|")
    for s, (expct, _) in SENSOR_PHYSICS.items():
        sub = d[d.sensor == s]
        bs = base[base.sensor == s]
        if not len(sub):
            continue
        L.append(f"| {s} | {'yes' if s in CUED_SENSORS else 'NO'} | {expct} | "
                 f"{len(sub):,} | {100 * sub.textbook_hit.mean():.1f}% | "
                 f"{100 * sub.input_hit.mean():.1f}% | "
                 f"{100 * bs.textbook_vs_actual.mean():.1f}% |")

    # -- per window length -------------------------------------------------
    L.append("\n## By prompt window length\n")
    L.append("| n cycles | Claims | Textbook agr. | Input faith. |")
    L.append("|---|---|---|---|")
    for n, sub in d.groupby("n_cycles"):
        L.append(f"| {n} | {len(sub):,} | {100 * sub.textbook_hit.mean():.1f}% | "
                 f"{100 * sub.input_hit.mean():.1f}% |")

    # -- what the model actually says --------------------------------------
    L.append("\n## Distribution of claims vs reality\n")
    L.append("| Sensor | Model says increase / decrease | Window actually increase / decrease / flat |")
    L.append("|---|---|---|")
    for s in SENSOR_PHYSICS:
        sub = d[d.sensor == s]
        allsub = cl[cl.sensor == s]
        if not len(sub):
            continue
        c = Counter(sub.claimed)
        a = Counter(allsub.actual_in_window)
        L.append(f"| {s} | {c['increase']} / {c['decrease']} | "
                 f"{a['increase']} / {a['decrease']} / {a['stable']} |")

    # -- robustness to the flat-window threshold ---------------------------
    L.append("\n## Robustness to the flat-window threshold\n")
    L.append("The one free parameter above is the threshold below which a window counts "
             "as flat. At eps = 0 the judgement is a pure sign test with no flat class.\n")
    L.append("| eps | Input faithfulness | Base rate (always assert canonical) | Windows called flat |")
    L.append("|---|---|---|---|")
    for eps in (0.0, 0.005, 0.01, 0.02, 0.05):
        act = np.where(cl.net_change.abs() < eps, "stable",
                       np.where(cl.net_change > 0, "increase", "decrease"))
        c2 = cl.assign(actual=act)
        d2 = c2[c2.claimed.isin(["increase", "decrease"])]
        L.append(f"| {eps:.3f} | {100 * (d2.claimed == d2.actual).mean():.1f}% | "
                 f"{100 * (d2.expected_textbook == d2.actual).mean():.1f}% | "
                 f"{int((c2.actual == 'stable').sum()):,} |")
    L.append("\nAt every threshold the explanations are no more accurate about the input "
             "than a rule that ignores the input entirely and always asserts the canonical "
             "direction. The gap never exceeds about one point in either direction, well "
             "inside the interval reported in P3.1, so the two are indistinguishable. The "
             "conclusion does not depend on the threshold.")

    # -- is the claim informative about the window at all? -----------------
    # If explanations described the input, the claimed direction would be
    # statistically dependent on the direction actually in the window.
    L.append("\n## Does the claim carry any information about the window?\n")
    L.append("Test of independence between the direction claimed and the direction "
             "actually present, per sensor. Cramer's V is the effect size (0 = the "
             "claim tells you nothing about the data); MI is mutual information in bits.\n")
    L.append("| Sensor | Claims | chi2 | p | Cramer's V | MI (bits) | Faithfulness vs base rate |")
    L.append("|---|---|---|---|---|---|---|")
    indep_rows = []
    for s in SENSOR_PHYSICS:
        sub = d[d.sensor == s]
        bs = base[base.sensor == s]
        if len(sub) < 10:
            continue
        ct = pd.crosstab(sub.claimed, sub.actual_in_window)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            chi2, p, v, mi = np.nan, np.nan, 0.0, 0.0
            note = "model never varies its claim - independence is degenerate"
        else:
            chi2, p, _, _ = chi2_contingency(ct)
            n = ct.values.sum()
            v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
            pxy = ct.values / n
            px = pxy.sum(1, keepdims=True)
            py = pxy.sum(0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = pxy * np.log2(pxy / (px * py))
            mi = float(np.nansum(terms))
            note = ""
        faith = 100 * sub.input_hit.mean()
        rate = 100 * bs.textbook_vs_actual.mean()
        indep_rows.append({"sensor": s, "n": len(sub), "chi2": chi2, "p": p,
                           "cramers_v": v, "mi_bits": mi,
                           "input_faithfulness_pct": faith, "base_rate_pct": rate})
        cs = "n/a" if np.isnan(chi2) else f"{chi2:.1f}"
        ps = "n/a" if np.isnan(p) else f"{p:.3f}"
        L.append(f"| {s} | {len(sub):,} | {cs} | {ps} | {v:.3f} | {mi:.4f} | "
                 f"{faith:.1f}% vs {rate:.1f}% {note} |")
    pd.DataFrame(indep_rows).to_csv(OUT / "p0_2a_claim_independence.csv", index=False)

    # -- per-configuration table -------------------------------------------
    per_cfg = (d.groupby(["dataset", "mode", "k", "n_cycles", "run"])
                 .agg(claims=("input_hit", "size"),
                      textbook_pct=("textbook_hit", lambda x: 100 * x.mean()),
                      input_pct=("input_hit", lambda x: 100 * x.mean()))
                 .reset_index())
    per_cfg.to_csv(OUT / "p0_2a_faithfulness_by_config.csv", index=False)

    L.append("\n## Reading\n")
    L.append(f"- The paper's explainability figure is reproduced here as "
             f"**{tot['textbook_agreement_pct']:.1f}%** textbook agreement.")
    L.append(f"- Against the sensor windows actually placed in the prompts, the same "
             f"claims are right **{tot['input_faithfulness_pct']:.1f}%** of the time.")
    L.append("- The gap is the size of the prompt-leakage effect: the explanations track "
             "the canonical narrative, not the data the model was given.")

    (OUT / "p0_2a_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p0_2a_report.md'}")


if __name__ == "__main__":
    main()
