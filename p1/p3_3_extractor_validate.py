"""
P3.3b - Score the claim extractor against the author's reading.

The labels come from results_p1/p3_3_author_labels.csv, written by
p3_3_import_readings.py from the author's reading. The reader saw, in shuffled
order, the full response for each (response, sensor) pair with every occurrence of
the sensor's name highlighted, the values of that sensor in the prompt and the
whole prompt, but never the extractor's verdict. The script refuses to run until
every case carries a label, so no figure below can come from anything else.

The sample is stratified over the extractor's own verdict, so the raw accuracy in
the sample is not the accuracy in the corpus. Every corpus-level figure is
reweighted by the true frequency of each verdict.

Nothing qualitative is asserted here that the labels do not show. The error modes
are counted and listed case by case, so the descriptions in the manuscript can be
checked against the cases themselves.

Outputs:
  results_p1/p3_3_extractor_sample.csv   the sample with the author's labels
  results_p1/p3_3_extractor_report.md
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
P0, P1 = EXP / "results_p0", EXP / "results_p1"
LABELS_FILE = P1 / "p3_3_author_labels.csv"
SAMPLE = P1 / "p3_3_extractor_sample.csv"
REPORT = P1 / "p3_3_extractor_report.md"

# frequency of each extractor verdict over the whole corpus, from p3_3_extractor_sample.py
CORPUS = {"absent": 13451, "decrease": 1956, "increase": 1860,
          "unspecified": 1624, "stable": 9}
LABELS = ["increase", "decrease", "stable", "unspecified", "absent"]
VALID = set(LABELS) | {"ambiguous"}
DIRN = ["increase", "decrease"]
REFERENCE = {"s4": "increase", "s7": "decrease", "s9": "decrease", "s11": "increase",
             "s12": "increase", "s14": "decrease", "s15": "increase"}
# the corpus figures the manuscript reports, for comparison with the rescored sample
MANUSCRIPT = {"ref": 91.3, "window": 42.7, "base": 42.0, "bench": 47.4, "conflict_ref": 89.5}
BOOT, SEED = 10000, 20260915


def read_author_labels():
    """The author's labels, keyed by sample id."""
    if not LABELS_FILE.exists():
        raise SystemExit(f"missing {LABELS_FILE.name}: run p1/p3_3_import_readings.py")
    lab = pd.read_csv(LABELS_FILE, keep_default_na=False)
    empty = lab[lab.author.str.strip() == ""].id.tolist()
    if empty:
        raise SystemExit(f"The reading is not complete: {len(empty)} of {len(lab)} cases have "
                         "no label. Nothing is scored until every case carries the author's "
                         "label.")
    bad = lab[~lab.author.isin(VALID)]
    if len(bad):
        raise SystemExit(f"labels outside the six allowed ones: "
                         f"{bad[['id', 'author']].to_dict('records')}")
    return lab.rename(columns={"sensor": "sensor_sheet"})


def window_direction(sample):
    """The direction actually in the supplied window, from the claims file."""
    claims = pd.read_csv(P0 / "p0_2a_claims.csv")
    out = []
    for _, r in sample.iterrows():
        stem = Path(r.file).stem.replace("traces_", "")
        ds, rest = stem[:5], stem[6:]
        m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
        key = dict(dataset=ds, mode=m.group(1), k=int(m.group(2)),
                   n_cycles=int(m.group(3)) if m.group(3) else 5,
                   run=m.group(4) or "run1", engine_id=int(r.engine_id), sensor=r.sensor)
        hit = claims
        for c, v in key.items():
            hit = hit[hit[c] == v]
        out.append(hit.actual_in_window.iloc[0] if len(hit) == 1 else None)
    return out


def attach_directions(d):
    """The three directions every sampled pair can be scored against: the reference, the
    window it was shown, and the benchmark of its own sub-dataset where that attests one."""
    d = d.copy()
    d["window"] = window_direction(d)
    if d.window.isna().any():
        raise SystemExit(f"no window direction for ids {d[d.window.isna()].id.tolist()}")
    d["ref"] = d.sensor.map(REFERENCE)
    d["dataset"] = d.file.str.extract(r"traces_(FD00\d)_")[0]
    bench = pd.read_csv(P1 / "p3_5_directions.csv").set_index(["dataset", "sensor"])
    d["attested"] = [bool(bench.loc[(r.dataset, r.sensor), "attested"]) for r in d.itertuples()]
    d["measured"] = [bench.loc[(r.dataset, r.sensor), "measured_direction"]
                     for r in d.itertuples()]
    d["conflict"] = d.attested & (d.measured != d.ref)
    return d


FIGURES = [("claims", "Directional claims in the corpus"),
           ("ref", "Agreement with the reference direction"),
           ("window", "Faithfulness to the supplied window"),
           ("base", "Base rate on the same claims"),
           ("gap", "Faithfulness minus base rate"),
           ("bench", "Agreement with the empirical benchmark direction, scorable claims"),
           ("conflict_ref", "Conflict set: share stating the reference direction")]


def indicators(d, col):
    """Per-pair numerators and denominators of each figure, if the pair is labelled by `col`."""
    lab = d[col].to_numpy()
    dirn = np.isin(lab, DIRN)
    ref, win = d.ref.to_numpy(), d.window.to_numpy()
    att, conf = d.attested.to_numpy(), d.conflict.to_numpy()
    meas = d.measured.to_numpy()
    return {"claims": dirn, "ref": dirn & (lab == ref), "window": dirn & (lab == win),
            "base": dirn & (ref == win), "scorable": dirn & att,
            "bench": dirn & att & (lab == meas), "conflict": dirn & conf,
            "conflict_ref": dirn & conf & (lab == ref)}


def figures(ind, w, idx):
    """Corpus-level figures from the pairs `idx`, each weighted by its stratum's corpus
    frequency over its sampled size (Horvitz-Thompson for a stratified sample)."""
    s = {k: float((w[idx] * v[idx]).sum()) for k, v in ind.items()}
    c = s["claims"]
    return {"claims": c, "ref": 100 * s["ref"] / c, "window": 100 * s["window"] / c,
            "base": 100 * s["base"] / c, "gap": 100 * (s["window"] - s["base"]) / c,
            "bench": 100 * s["bench"] / s["scorable"] if s["scorable"] else np.nan,
            "conflict_ref": (100 * s["conflict_ref"] / s["conflict"]
                             if s["conflict"] else np.nan)}


def main():
    lab = read_author_labels()
    d = pd.read_csv(SAMPLE)
    if "human" in d.columns:
        raise SystemExit("the sample still carries a `human` column from an earlier labelling; "
                         "it is not the author's reading and must not be scored")
    d = d.drop(columns=[c for c in ("author", "author_note") if c in d.columns])
    d = d.merge(lab[["id", "sensor_sheet", "author", "author_note"]], on="id", how="left")
    mism = d[d.sensor != d.sensor_sheet]
    if len(mism):
        raise SystemExit(f"sheet and sample disagree on the sensor for ids {mism.id.tolist()}")
    d = d.drop(columns="sensor_sheet")
    d.to_csv(SAMPLE, index=False)

    n_total = len(d)
    amb = d[d.author == "ambiguous"]
    scored = attach_directions(d[d.author != "ambiguous"])
    cm = pd.crosstab(scored.extractor, scored.author).reindex(
        index=LABELS, columns=LABELS, fill_value=0)

    L = ["# P3.3 - Validating the directional-claim extractor\n"]
    L.append(f"A stratified sample of {n_total} (response, sensor) pairs was labelled by the "
             "author, in shuffled order, without the extractor's verdict. Each case showed the "
             "full response with every occurrence of the sensor's name highlighted, a chart of "
             "the values of that sensor in the prompt, and the whole prompt on request. When "
             "the sensor's name did not occur in the response the screen said so, so for those "
             "cases the reader's task was to check whether the sensor was referred to in some "
             "other form. Each pair was labelled increase, decrease, stable, unspecified, "
             "absent or ambiguous following the guideline in `p3_3_reading_guideline.md`. The "
             "sample is stratified over the extractor's own verdict so that both kinds of error "
             "can be measured; the raw accuracy in the sample is therefore not the accuracy in "
             "the corpus, and the corpus figures below are reweighted by how often each verdict "
             "occurs.\n")

    L.append(f"\n## Ambiguous cases\n\n{len(amb)} of {n_total} were labelled ambiguous and are "
             "excluded from the scoring. They are listed so that their kind can be seen:\n")
    for _, r in amb.iterrows():
        L.append(f"- id {r.id}, {r.sensor}, extractor {r.extractor}: "
                 f"\"{str(r.context)[:220]}\"" + (f" (note: {r.author_note})"
                                                   if r.author_note else ""))

    L.append("\n## Confusion matrix, sampled cases\n")
    L.append("Rows are the extractor, columns the author's reading.\n")
    L.append("| extractor \\ author | " + " | ".join(LABELS) + " | total |")
    L.append("|---" * (len(LABELS) + 2) + "|")
    for lab_ in LABELS:
        row = cm.loc[lab_]
        L.append(f"| **{lab_}** | " + " | ".join(str(int(v)) for v in row)
                 + f" | {int(row.sum())} |")

    L.append("\n## Precision and recall, sampled cases\n")
    L.append("| Class | Precision | Recall | Sampled n (author) |")
    L.append("|---|---|---|---|")
    for lab_ in LABELS:
        tp = int(cm.loc[lab_, lab_])
        fp = int(cm.loc[lab_].sum() - tp)
        fn = int(cm[lab_].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        L.append(f"| {lab_} | {prec:.3f} | {rec:.3f} | {int(cm[lab_].sum())} |")

    tp_d = int(cm.loc[DIRN, DIRN].values.sum())
    fp_d = int(cm.loc[DIRN].values.sum() - tp_d)
    fn_d = int(cm[DIRN].values.sum() - tp_d)
    prec_d, rec_d = tp_d / (tp_d + fp_d), tp_d / (tp_d + fn_d)
    sign_ok = int(cm.loc["increase", "increase"] + cm.loc["decrease", "decrease"])
    L.append(f"\nTreating increase and decrease together as **a directional claim**, "
             f"precision is **{prec_d:.3f}** and recall **{rec_d:.3f}** in the sample. Of the "
             f"{tp_d} cases where both the extractor and the author see a directional claim, "
             f"the extractor gets the sign right in {sign_ok}, "
             f"**{100 * sign_ok / tp_d:.1f}%**.\n")

    L.append("\n## Reweighted to the corpus\n")
    L.append("| Extractor verdict | Corpus count | Sampled | Agrees with the author | "
             "Implied corpus accuracy |")
    L.append("|---|---|---|---|---|")
    tot = sum(CORPUS.values())
    weighted = 0.0
    for lab_ in LABELS:
        sub = scored[scored.extractor == lab_]
        if not len(sub):
            L.append(f"| {lab_} | {CORPUS[lab_]:,} | 0 | n/a | n/a |")
            continue
        acc = (sub.author == lab_).mean()
        weighted += CORPUS[lab_] * acc
        L.append(f"| {lab_} | {CORPUS[lab_]:,} | {len(sub)} | "
                 f"{int((sub.author == lab_).sum())} | {100 * acc:.1f}% |")
    L.append(f"\nWeighting each verdict by how often it occurs, the extractor agrees with the "
             f"author on **{100 * weighted / tot:.1f}%** of all {tot:,} pairs. The absent "
             f"class carries {100 * CORPUS['absent'] / tot:.0f}% of that weight, which is why "
             "each of its sampled responses was read in full. Since the screen said when the "
             "sensor's name did not occur, agreement on that class measures only whether the "
             "sensor was referred to in some other form.\n")

    # The analysis uses one distinction only: whether a pair carries an increase, a
    # decrease, or no directional claim. Stable, unspecified and absent all count as
    # no claim, so a disagreement between them moves no figure in the paper.
    three = lambda s: s.where(s.isin(DIRN), "none")  # noqa: E731
    w3 = 0.0
    L.append("\nThe analysis itself uses one distinction only: whether a pair carries an "
             "increase, a decrease, or no directional claim. Stable, unspecified and absent "
             "all enter it as no claim, so a disagreement among them moves no figure in the "
             "paper. On that three-way distinction:\n")
    L.append("| Extractor verdict | Sampled | Agrees with the author | Implied corpus accuracy |")
    L.append("|---|---|---|---|")
    for lab_ in LABELS:
        sub = scored[scored.extractor == lab_]
        if not len(sub):
            continue
        ok = (three(sub.extractor) == three(sub.author))
        w3 += CORPUS[lab_] * ok.mean()
        L.append(f"| {lab_} | {len(sub)} | {int(ok.sum())} | {100 * ok.mean():.1f}% |")
    L.append(f"\nWeighted in the same way, the extractor and the author agree on whether a "
             f"pair carries a directional claim, and on its sign, for **{100 * w3 / tot:.1f}%** "
             f"of all {tot:,} pairs.\n")

    # ---- the three error modes, counted and listed -------------------------
    # Missed claims are scaled stratum by stratum: the stable stratum is a census and
    # the unspecified one a sample, so a pooled rate would mix the two.
    L.append("\n## Missed claims\n")
    nodir = scored[scored.extractor.isin(["unspecified", "stable", "absent"])]
    miss = nodir[nodir.author.isin(DIRN)]
    est, parts = 0.0, []
    for lab_ in ("unspecified", "stable", "absent"):
        sub = nodir[nodir.extractor == lab_]
        k = int(sub.author.isin(DIRN).sum())
        est += CORPUS[lab_] * k / len(sub)
        parts.append(f"{k} of {len(sub)} called {lab_}")
    no_claim = CORPUS["unspecified"] + CORPUS["stable"]
    on_ref = int((miss.author == miss.ref).sum())
    faithful = int((miss.author == miss.window).sum())
    L.append(f"{len(miss)} sampled cases the extractor counted as carrying no directional claim "
             f"carry one in the author's reading ({', '.join(parts)}). Scaled stratum by "
             f"stratum that is roughly **{est:,.0f}** claims, **{100 * est / no_claim:.0f}%** "
             f"of the {no_claim:,} pairs called unspecified or stable, not counted against the "
             f"3,816 that were. Of the {len(miss)}, **{on_ref}** state the reference direction "
             f"and {faithful} match the direction in the supplied window. The cases:\n")
    for _, r in miss.iterrows():
        L.append(f"- id {r.id}, {r.sensor}, extractor {r.extractor}, author {r.author}, "
                 f"window {r.window}: \"{str(r.context)[:220]}\"")

    L.append("\n## Directional calls the author did not count\n")
    calls = scored[scored.extractor.isin(DIRN)]
    extra = calls[~calls.author.isin(DIRN)]
    est_x = sum(CORPUS[v] * (~calls[calls.extractor == v].author.isin(DIRN)).mean()
                for v in DIRN)
    L.append(f"Of the {len(calls)} sampled directional calls, **{len(extra)}** carry no "
             f"directional claim in the author's reading ("
             + ", ".join(f"{n} read as {k}" for k, n in extra.author.value_counts().items())
             + f"), which scales to roughly **{est_x:,.0f}** of the 3,816 claims counted. Of "
             f"the {len(extra)}, **{int((extra.extractor == extra.ref).sum())}** carry the "
             f"reference direction and {int((extra.extractor == extra.window).sum())} match "
             "the direction in the supplied window. The cases:\n")
    for _, r in extra.iterrows():
        L.append(f"- id {r.id}, {r.sensor}, extractor {r.extractor}, author {r.author}, "
                 f"window {r.window}: \"{str(r.context)[:220]}\"")

    L.append("\n## Wrong sign\n")
    both = scored[scored.extractor.isin(DIRN) & scored.author.isin(DIRN)].copy()
    wrong = both[both.extractor != both.author].copy()
    L.append(f"Of the {len(calls)} sampled directional calls, **{len(wrong)}** carry the "
             f"opposite sign to the author's reading, one in "
             f"{len(calls) / max(len(wrong), 1):.0f}.")
    if len(wrong):
        up = int((wrong.author == wrong.window).sum())
        down = int((wrong.extractor == wrong.window).sum())
        L.append(f"Correcting them would make {up} faithful to the window and {down} "
                 f"unfaithful, with {len(wrong) - up - down} on a flat window. "
                 + ("The errors therefore do not lean towards making the model look less "
                    "faithful than it is." if up <= down else
                    "On balance they make the model look less faithful than it is, which "
                    "works against the paper's argument and has to be stated."))
        L.append("\nThe cases:\n")
        for _, r in wrong.iterrows():
            L.append(f"- id {r.id}, {r.sensor}, extractor {r.extractor}, author {r.author}, "
                     f"window {r.window}: \"{str(r.context)[:220]}\"")

    # ---- what the three errors together do to the figures ------------------
    # Both readings are scored on the same sampled pairs with the same weights, so the
    # difference isolates the extractor's errors from the sample's own sampling error.
    L.append("\n## What the errors do to the faithfulness figures\n")
    w = np.array([CORPUS[v] / int((scored.extractor == v).sum()) for v in scored.extractor])
    ind_e, ind_a = indicators(scored, "extractor"), indicators(scored, "author")
    all_idx = np.arange(len(scored))
    fe, fa = figures(ind_e, w, all_idx), figures(ind_a, w, all_idx)
    strata = [np.flatnonzero(scored.extractor.to_numpy() == v) for v in CORPUS]
    rng = np.random.default_rng(SEED)
    diff = {k: np.empty(BOOT) for k, _ in FIGURES}
    for b in range(BOOT):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata if len(s)])
        e, a = figures(ind_e, w, idx), figures(ind_a, w, idx)
        for k, _ in FIGURES:
            diff[k][b] = a[k] - e[k]
    L.append("Each figure is estimated twice from the same sample, once with the extractor's "
             "verdicts and once with the author's labels, weighting every stratum by its "
             "corpus frequency. The first column is therefore the sample's own estimate of "
             "what the corpus shows, and the difference is what the extractor's errors do to "
             "it. Intervals are 95% percentile intervals from "
             f"{BOOT:,} bootstrap resamples within strata, paired between the two readings.\n")
    L.append("| Figure | Manuscript (corpus) | Extractor, sample | Author, sample | "
             "Difference | 95% interval |")
    L.append("|---|---|---|---|---|---|")
    for k, name in FIGURES:
        lo, hi = np.nanpercentile(diff[k], [2.5, 97.5])
        if k == "claims":
            L.append(f"| {name} | 3,816 | {fe[k]:,.0f} | {fa[k]:,.0f} | {fa[k] - fe[k]:+,.0f} | "
                     f"{lo:+,.0f} to {hi:+,.0f} |")
            continue
        man = f"{MANUSCRIPT[k]:.1f}%" if k in MANUSCRIPT else (
            f"{MANUSCRIPT['window'] - MANUSCRIPT['base']:+.1f}" if k == "gap" else "")
        unit = "" if k == "gap" else "%"
        L.append(f"| {name} | {man} | {fe[k]:.1f}{unit} | {fa[k]:.1f}{unit} | "
                 f"{fa[k] - fe[k]:+.1f} | {lo:+.1f} to {hi:+.1f} |")
    a_dir = scored[scored.author.isin(DIRN)]
    e_off = calls[calls.extractor != calls.ref]
    a_conf = a_dir[a_dir.conflict]
    L.append(f"\nIn the author's reading **{int((a_dir.author == a_dir.ref).sum())} of the "
             f"{len(a_dir)}** sampled directional claims state the reference direction, "
             f"including {int((a_conf.author == a_conf.ref).sum())} of the {len(a_conf)} about "
             "a sensor whose own sub-dataset has it the other way round. Of the "
             f"{len(e_off)} sampled calls in which the extractor finds a departure from the "
             f"reference, the author reads {int((e_off.author == e_off.extractor).sum())} the "
             "same way. Where every claim states the reference direction, faithfulness to the "
             "window equals the base rate by construction.\n")
    L.append("The clauses about deviations, variance or levels, which the extractor counts and "
             "the guideline does not, are concentrated where the prompt names no direction; "
             "their effect on the cue comparisons of Section 4.4 is computed in "
             "`p3_3_extractor_sensitivity.md`.\n")

    L.append("\n## Limits of this validation\n")
    L.append("One reader, the author, so no inter-annotator statistic is available and the "
             "numbers measure the extractor against one careful reading. The reading is the one "
             "step of the analysis that cannot be rerun from the code: a replication has to "
             "repeat it, on the released sample and with the released guideline. The sample is "
             "stratified rather than random, so only the reweighted figures describe the "
             "corpus, and with 80 sampled directional calls every corpus-level estimate here "
             "carries the sampling error shown in the intervals.\n")

    # ---- every figure the manuscript quotes, for audit_numbers.py ----------
    from p3_3_extractor_sensitivity import OFF_TREND, matched_clause, traces_by_key
    texts = traces_by_key()

    def clause(r):
        stem = r.file.replace("traces_", "").replace(".json", "")
        ds, rest = stem[:5], stem[6:]
        m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
        return matched_clause(texts[(ds, m.group(1), int(m.group(2)),
                                     int(m.group(3)) if m.group(3) else 5,
                                     m.group(4) or "run1", int(r.engine_id))], r.sensor)
    us = scored[scored.extractor.isin(["unspecified", "stable"])]
    absent = scored[scored.extractor == "absent"]
    fig = {"pairs read": n_total, "ambiguous": len(amb),
           "precision, directional": prec_d, "recall, directional": rec_d,
           "both see a claim": tp_d, "same sign": sign_ok,
           "corpus agreement, full label %": 100 * weighted / tot,
           "corpus agreement, claim and sign %": 100 * w3 / tot,
           "absent sampled": len(absent),
           "absent read otherwise": int((absent.author != "absent").sum()),
           "unspecified or stable sampled": len(us),
           "missed among unspecified or stable": int(us.author.isin(DIRN).sum()),
           "missed, corpus estimate": est, "missed stating reference": on_ref,
           "directional calls sampled": len(calls), "calls not counted by author": len(extra),
           "of which deviation, variance or level wording":
               int(sum(bool(OFF_TREND.search(clause(r))) for r in extra.itertuples())),
           "wrong sign": len(wrong),
           "author claims": len(a_dir),
           "author claims stating reference": int((a_dir.author == a_dir.ref).sum()),
           "author conflict claims": len(a_conf),
           "author conflict claims stating reference": int((a_conf.author == a_conf.ref).sum()),
           "extractor departures from reference": len(e_off),
           "extractor departures the author confirms":
               int((e_off.author == e_off.extractor).sum())}
    for k, _ in FIGURES:
        lo, hi = np.nanpercentile(diff[k], [2.5, 97.5])
        fig.update({f"rescored {k}, extractor": fe[k], f"rescored {k}, author": fa[k],
                    f"rescored {k}, difference": fa[k] - fe[k],
                    f"rescored {k}, CI lo": lo, f"rescored {k}, CI hi": hi})
    pd.Series(fig, name="value").rename_axis("figure").to_csv(P1 / "p3_3_extractor_figures.csv")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {REPORT}")

    # Optional, for the author only: where this reading differs from the provisional
    # labels that preceded it. Off unless PROVISIONAL_LABELS names that file; not
    # released and not used for any figure.
    import os
    prov = Path(os.environ.get("PROVISIONAL_LABELS", "")) if os.environ.get(
        "PROVISIONAL_LABELS") else None
    if prov is not None and prov.exists():
        p = d.merge(pd.read_csv(prov), on="id")
        diff = p[p.author != p.provisional]
        out = [f"# Dove la lettura dell'autore differisce dalle etichette provvisorie\n",
               f"{len(p) - len(diff)} casi su {len(p)} coincidono. Solo per consultazione: "
               "i numeri del paper vengono dalla lettura dell'autore.\n"]
        for _, r in diff.iterrows():
            out.append(f"- id {r.id}, {r.sensor}: autore **{r.author}**, provvisoria "
                       f"{r.provisional} — \"{str(r.context)[:200]}\"")
        (ROOT / "documenti" / "confronto_letture_estrattore.md").write_text(
            "\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
