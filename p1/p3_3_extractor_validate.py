"""
P3.3b - Score the claim extractor against the hand-annotated sample.

Reads results_p1/p3_3_extractor_sample.csv, which carries the extractor's verdict
and a human verdict for each sampled (response, sensor) pair, and reports a
confusion matrix, per-class precision and recall, and the effect of the errors on
the two headline percentages.

The sample is stratified over the extractor's own verdict, so the raw accuracy in
the sample is not the accuracy in the corpus. Every corpus-level figure below is
reweighted by the true frequency of each verdict.

Annotation was done by one annotator against the local context that the extractor
itself reads. There is no second annotator, so no inter-rater agreement is
reported and the exercise measures the extractor against one careful reading
rather than against a consensus.

Output: results_p1/p3_3_extractor_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P1 = EXP / "results_p1"

# frequency of each extractor verdict over the whole corpus, from p3_3_extractor_sample.py
CORPUS = {"absent": 13451, "decrease": 1956, "increase": 1860,
          "unspecified": 1624, "stable": 9}
LABELS = ["increase", "decrease", "stable", "unspecified", "absent"]


def main():
    d = pd.read_csv(P1 / "p3_3_extractor_sample.csv")
    d = d[d.human.notna() & (d.human.astype(str).str.strip() != "")].copy()
    if d.empty:
        raise SystemExit("no annotations found in the `human` column")
    n_total = len(d)
    amb = d[d.human == "ambiguous"]
    scored = d[d.human != "ambiguous"].copy()

    L = ["# P3.3 - Validating the directional-claim extractor\n"]
    L.append(f"A stratified sample of {n_total} (response, sensor) pairs was read by hand "
             "against the same local context the extractor sees, and labelled increase, "
             "decrease, stable, unspecified or absent. The sample is stratified over the "
             "extractor's own verdict so that both kinds of error can be measured, which "
             "means the raw accuracy in the sample is not the accuracy in the corpus; the "
             "corpus figures below are reweighted by how often each verdict actually "
             "occurs.\n")
    L.append(f"**{len(amb)} of {n_total} cases were labelled ambiguous** and are excluded "
             "from the scoring. They are all of one kind: the model hedges, as in "
             "\"relatively stable or decreasing values\", and no single label is correct. "
             "The extractor resolves such phrases inconsistently, calling the same "
             "construction stable in one response and decrease in another.\n")

    # ---- confusion matrix -------------------------------------------------
    L.append("\n## Confusion matrix, sampled cases\n")
    cm = pd.crosstab(scored.extractor, scored.human).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    L.append("Rows are the extractor, columns the human reading.\n")
    L.append("| extractor \\ human | " + " | ".join(LABELS) + " | total |")
    L.append("|---" * (len(LABELS) + 2) + "|")
    for lab in LABELS:
        row = cm.loc[lab]
        L.append(f"| **{lab}** | " + " | ".join(str(int(v)) for v in row)
                 + f" | {int(row.sum())} |")

    # ---- per class --------------------------------------------------------
    L.append("\n## Precision and recall, sampled cases\n")
    L.append("| Class | Precision | Recall | F1 | Sampled n (human) |")
    L.append("|---|---|---|---|---|")
    rows = []
    for lab in LABELS:
        tp = int(cm.loc[lab, lab])
        fp = int(cm.loc[lab].sum() - tp)
        fn = int(cm[lab].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
        rows.append((lab, prec, rec, f1, int(cm[lab].sum())))
        L.append(f"| {lab} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {int(cm[lab].sum())} |")

    # directional = increase or decrease, the class the analysis depends on
    dirn = ["increase", "decrease"]
    tp_d = int(cm.loc[dirn, dirn].values.sum())
    fp_d = int(cm.loc[dirn].values.sum() - tp_d)
    fn_d = int(cm[dirn].values.sum() - tp_d)
    prec_d = tp_d / (tp_d + fp_d)
    rec_d = tp_d / (tp_d + fn_d)
    # within the directional calls that are correct as "directional", how many
    # have the right sign?
    sign_ok = int(cm.loc["increase", "increase"] + cm.loc["decrease", "decrease"])
    sign_tot = int(cm.loc[dirn, dirn].values.sum())
    L.append(f"\nTreating increase and decrease together as **a directional claim**, which is "
             f"the distinction the faithfulness analysis rests on, precision is "
             f"{prec_d:.3f} and recall {rec_d:.3f} in the sample. Of the "
             f"{sign_tot} cases where both the extractor and the reader see a directional "
             f"claim, the extractor gets the sign right in {sign_ok}, "
             f"{100 * sign_ok / sign_tot:.1f}%.\n")

    # ---- corpus-level reweighting ----------------------------------------
    L.append("\n## Reweighted to the corpus\n")
    tot_corpus = sum(CORPUS.values())
    weighted_correct = 0.0
    L.append("| Extractor verdict | Corpus count | Sampled | Correct in sample | "
             "Implied corpus accuracy |")
    L.append("|---|---|---|---|---|")
    for lab in LABELS:
        sub = scored[scored.extractor == lab]
        if len(sub) == 0:
            L.append(f"| {lab} | {CORPUS[lab]:,} | 0 | n/a | n/a |")
            continue
        acc = (sub.human == lab).mean()
        weighted_correct += CORPUS[lab] * acc
        L.append(f"| {lab} | {CORPUS[lab]:,} | {len(sub)} | "
                 f"{int((sub.human == lab).sum())} | {100 * acc:.1f}% |")
    L.append(f"\nWeighting each verdict by how often it occurs, the extractor agrees with the "
             f"reader on **{100 * weighted_correct / tot_corpus:.1f}%** of all "
             f"{tot_corpus:,} (response, sensor) pairs in the corpus.\n")

    # ---- what the errors do to the headline numbers ----------------------
    L.append("\n## What the errors do to the headline percentages\n")
    inc_rate = (scored[scored.extractor == "increase"].human == "increase").mean()
    dec_rate = (scored[scored.extractor == "decrease"].human == "decrease").mean()
    miss = scored[(scored.extractor.isin(["unspecified", "stable"]))
                  & (scored.human.isin(dirn))]
    miss_rate = len(miss) / max(len(scored[scored.extractor.isin(["unspecified", "stable"])]), 1)
    est_missed = miss_rate * (CORPUS["unspecified"] + CORPUS["stable"])
    L.append(f"Two error modes matter and they pull in opposite directions.\n")
    L.append(f"- **Missed claims.** {len(miss)} of the "
             f"{len(scored[scored.extractor.isin(['unspecified', 'stable'])])} sampled cases "
             f"the extractor called unspecified or stable do carry a directional claim, "
             f"{100 * miss_rate:.0f}%. Scaled to the corpus that is roughly "
             f"{est_missed:,.0f} claims not counted, against the 3,816 that were. Almost all "
             "are the same construction: the direction is stated once for a list of sensors "
             "and the 60-character window before the sensor token does not reach back to the "
             "verb.")
    L.append(f"- **Wrong sign.** Of the sampled cases the extractor called increase, "
             f"{100 * inc_rate:.0f}% are read as increase; of those it called decrease, "
             f"{100 * dec_rate:.0f}% are read as decrease. The failure is a clause such as "
             "\"increasing values in s11 and s12, indicating a decline in performance\", "
             "where the trailing word belongs to the consequence and not to the sensor.\n")
    L.append("Both modes are conservative for the paper's argument rather than favourable to "
             "it. The missed claims are overwhelmingly canonical statements of the kind the "
             "prompt cues, so counting them would raise agreement with the textbook, which is "
             "already the high number. The sign errors are a few per cent and are not "
             "systematically aligned with the window, so they add noise to input faithfulness "
             "rather than bias. The gap between 91% and 43% is far larger than either.\n")

    L.append("\n## Limits of this validation\n")
    L.append("One annotator, no second reader, so no inter-rater statistic is available and "
             "the numbers measure the extractor against one careful reading. The annotator saw "
             "the extractor's verdict while labelling, which is a source of anchoring. The "
             "sample is stratified rather than random, so only the reweighted figures describe "
             "the corpus. The residual ambiguity is concentrated in two constructions: hedged "
             "disjunctions, and claims about a level or a variance rather than a trend, where "
             "\"values consistently lower than normal\" and \"increasing deviations\" are not "
             "statements about the direction of the series.\n")

    (P1 / "p3_3_extractor_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_3_extractor_report.md'}")


if __name__ == "__main__":
    main()
