"""
P3.3d - What the extractor's false directional calls do to the cue comparisons.

The reading guideline (results_p1/p3_3_reading_guideline.md) classes as unspecified
a mention whose level or variability is described without a direction, and the
author's reading of the validation sample (p3_3_extractor_validate.py) applied it to
every clause of that kind: "increasing deviations in efficiency sensors (s7 and
s8)", "increasing variance", "values consistently lower than normal", "lower
values". The extractor counts such a clause with the sign of the nearby word.
Such clauses are rare where the prompt names the sensor and common where it does
not, so they fall unevenly on the two sides of the comparisons in Section 4.4.

This script rebuilds, for every counted claim, the clause the extractor actually
matched, flags the clauses that use that wording, and recomputes the two cue
comparisons and the faithfulness figures with and without them, using the same
cluster bootstrap and cluster permutation as p3_1_cluster_inference.py. It is a
sensitivity analysis defined after the reading, and is reported as one: the
flagged wording is a lexical proxy for the reader's rule, not a new reading.

Outputs (results_p1/):
  p3_3_extractor_sensitivity.csv
  p3_3_extractor_sensitivity.md
"""
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(HERE))
from analyze_explainability import classify_mention  # noqa: E402
from p3_1_cluster_inference import (CUED, N_PERM, SEED, boot_ci,  # noqa: E402
                                    cluster_index, load_main, widest)

P1 = EXP / "results_p1"
# the wordings the reader did not count as a trend: each has at least one instance
# among the sampled pairs the author labelled unspecified (ids 28, 137: deviations;
# 130: variance; 102: lower than normal; 85: lower values), with fluctuation and
# higher values/than normal added as the same kind of statement
OFF_TREND = re.compile(r"deviation|variance|fluctuat|than normal|lower values|higher values")


def traces_by_key():
    out = {}
    for p in sorted((EXP / "results").glob("traces_*.json")):
        stem = p.stem.replace("traces_", "")
        ds, rest = stem[:5], stem[6:]
        m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
        key = (ds, m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else 5,
               m.group(4) or "run1")
        for t in json.loads(p.read_text(encoding="utf-8")):
            out[key + (int(t["engine_id"]),)] = t.get("reasoning", "") or ""
    return out


def matched_clause(text, sensor):
    """The text classify_mention looks at for this sensor, every mention joined:
    60 characters before the token and up to 80 after, cut at the next sentence
    or at the next sensor. Kept identical to analyze_explainability.classify_mention."""
    out = []
    for m in re.finditer(rf"\b{sensor}\b", text, re.IGNORECASE):
        start, end = max(0, m.start() - 60), min(len(text), m.end() + 80)
        after = text[m.end():end].lower()
        cut = re.search(r"[;.](\s|$)|,\s*s\d", after)
        out.append((text[start:m.end()] + (after[:cut.start()] if cut else after)).lower())
    return " || ".join(out)


def sample_check():
    """How the author labelled the sampled pairs whose matched clause uses the wording."""
    texts = traces_by_key()
    d = pd.read_csv(P1 / "p3_3_extractor_sample.csv", keep_default_na=False)
    if "author" not in d.columns:
        raise SystemExit("run p3_3_extractor_validate.py first: the sample has no labels yet")
    flagged = []
    for r in d.itertuples():
        stem = r.file.replace("traces_", "").replace(".json", "")
        ds, rest = stem[:5], stem[6:]
        m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
        key = (ds, m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else 5,
               m.group(4) or "run1", int(r.engine_id))
        if OFF_TREND.search(matched_clause(texts[key], r.sensor)):
            flagged.append(r.author)
    return len(flagged), sum(a == "unspecified" for a in flagged)


def contrasts(c, label, L, out):
    """The two cue comparisons of Section 4.4 and the faithfulness figures, on `c`."""
    tb = (c.claimed == c.expected_textbook).to_numpy(float)
    win = (c.claimed == c.actual_in_window).to_numpy(float)
    base = (c.actual_in_window == c.expected_textbook).to_numpy(float)
    cued = c.sensor.isin(CUED).to_numpy()
    zs = (c["mode"] == "zero_shot").to_numpy()
    sensor = c.sensor.to_numpy()
    rows = np.arange(len(c))
    _, eng = cluster_index(c.engine.to_numpy())
    cfg_codes, cfg = cluster_index(c.config.to_numpy())

    def diff(v, mask):
        def f(r, _m=mask):
            m = _m[r]
            if m.all() or (~m).all():
                return np.nan
            return v[r][m].mean() - v[r][~m].mean()
        return f

    L.append(f"\n## {label}\n")
    L.append(f"{len(c):,} claims: {int(cued.sum()):,} about the five sensors the zero-shot "
             f"template names, {int((~cued).sum())} about s4 and s7; {int(zs.sum()):,} from "
             f"zero-shot runs, {int((~zs).sum())} from few-shot runs.\n")
    L.append("| Contrast | Side A | Side B | Difference | 95% CI (cluster) | p (cluster) |")
    L.append("|---|---|---|---|---|---|")

    # 1. named sensors against s4 and s7: the permutation relabels sensors, as in p3_1
    obs = diff(tb, cued)(rows)
    lo, hi = widest(boot_ci(eng, diff(tb, cued)), boot_ci(cfg, diff(tb, cued)))
    sensors = np.array(sorted(set(sensor)))
    n_c = len(CUED & set(sensors))
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        fake = np.isin(sensor, rng.choice(sensors, size=n_c, replace=False))
        v = diff(tb, fake)(rows)
        hits += bool(np.isfinite(v) and abs(v) >= abs(obs) - 1e-12)
    p = (hits + 1) / (N_PERM + 1)
    L.append(f"| Reference agreement, named sensors vs s4/s7 | {100 * tb[cued].mean():.1f}% | "
             f"{100 * tb[~cued].mean():.1f}% | {100 * obs:.1f} | [{100 * lo:.1f}, "
             f"{100 * hi:.1f}] | {p:.4f} (floor {1 / comb(len(sensors), n_c):.3f}) |")
    out.append(dict(set=label, contrast="reference agreement, named vs uncued sensors",
                    a=100 * tb[cued].mean(), b=100 * tb[~cued].mean(), estimate=100 * obs,
                    ci_lo=100 * lo, ci_hi=100 * hi, p=p, n_a=int(cued.sum()),
                    n_b=int((~cued).sum())))

    # 2. zero-shot template against few-shot: the permutation relabels configurations
    obs = diff(tb, zs)(rows)
    lo, hi = widest(boot_ci(eng, diff(tb, zs)), boot_ci(cfg, diff(tb, zs)))
    lab = pd.Series(zs).groupby(cfg_codes).first().to_numpy().copy()
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(lab)
        v = diff(tb, lab[cfg_codes])(rows)
        hits += bool(np.isfinite(v) and abs(v) >= abs(obs) - 1e-12)
    p = (hits + 1) / (N_PERM + 1)
    L.append(f"| Reference agreement, zero-shot vs few-shot | {100 * tb[zs].mean():.1f}% | "
             f"{100 * tb[~zs].mean():.1f}% | {100 * obs:.1f} | [{100 * lo:.1f}, "
             f"{100 * hi:.1f}] | {p:.4f} |")
    out.append(dict(set=label, contrast="reference agreement, zero-shot vs few-shot",
                    a=100 * tb[zs].mean(), b=100 * tb[~zs].mean(), estimate=100 * obs,
                    ci_lo=100 * lo, ci_hi=100 * hi, p=p, n_a=int(zs.sum()),
                    n_b=int((~zs).sum())))

    # 3. faithfulness on the two sides, and against the base rate overall
    obs = diff(win, cued)(rows)
    lo, hi = widest(boot_ci(eng, diff(win, cued)), boot_ci(cfg, diff(win, cued)))
    L.append(f"| Faithfulness to the window, named sensors vs s4/s7 | "
             f"{100 * win[cued].mean():.1f}% | {100 * win[~cued].mean():.1f}% | "
             f"{100 * obs:.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] | |")
    out.append(dict(set=label, contrast="faithfulness, named vs uncued sensors",
                    a=100 * win[cued].mean(), b=100 * win[~cued].mean(), estimate=100 * obs,
                    ci_lo=100 * lo, ci_hi=100 * hi, p=np.nan, n_a=int(cued.sum()),
                    n_b=int((~cued).sum())))

    # faithfulness against the base rate within each group: the two groups have
    # different base rates, so the raw difference above mixes the sensors' own
    # trends with anything the cue does
    for name, m in [("named sensors", cued), ("s4/s7", ~cued)]:
        def g(r, _m=m):
            mm = _m[r]
            return win[r][mm].mean() - base[r][mm].mean() if mm.any() else np.nan
        obs = g(rows)
        lo, hi = widest(boot_ci(eng, g), boot_ci(cfg, g))
        L.append(f"| Faithfulness vs base rate, {name} | {100 * win[m].mean():.1f}% | "
                 f"{100 * base[m].mean():.1f}% | {100 * obs:+.1f} | [{100 * lo:+.1f}, "
                 f"{100 * hi:+.1f}] | |")
        out.append(dict(set=label, contrast=f"faithfulness minus base rate, {name}",
                        a=100 * win[m].mean(), b=100 * base[m].mean(), estimate=100 * obs,
                        ci_lo=100 * lo, ci_hi=100 * hi, p=np.nan, n_a=int(m.sum()),
                        n_b=int(m.sum())))

    def gap(r):
        return win[r].mean() - base[r].mean()
    obs = gap(rows)
    lo, hi = widest(boot_ci(eng, gap), boot_ci(cfg, gap))
    L.append(f"| Faithfulness vs base rate, all claims | {100 * win.mean():.1f}% | "
             f"{100 * base.mean():.1f}% | {100 * obs:+.1f} | [{100 * lo:+.1f}, "
             f"{100 * hi:+.1f}] | |")
    out.append(dict(set=label, contrast="faithfulness minus base rate",
                    a=100 * win.mean(), b=100 * base.mean(), estimate=100 * obs,
                    ci_lo=100 * lo, ci_hi=100 * hi, p=np.nan, n_a=len(c), n_b=len(c)))
    L.append(f"\nAgreement with the reference direction over all {len(c):,} claims: "
             f"{100 * tb.mean():.1f}%.\n")


def main():
    c = load_main()
    texts = traces_by_key()
    keyed = [(r.dataset, r.mode, int(r.k), int(r.n_cycles), r.run, int(r.engine_id))
             for r in c.itertuples()]
    c["clause"] = [matched_clause(texts[k], s) for k, s in zip(keyed, c.sensor)]
    # the clause is only meaningful if it is the one the extractor classified
    same = [classify_mention(texts[k], s) == v for k, s, v in zip(keyed, c.sensor, c.claimed)]
    if not all(same):
        raise SystemExit(f"{len(same) - sum(same)} claims do not reproduce from their traces")
    c["off_trend"] = c.clause.str.contains(OFF_TREND)

    L = ["# P3.3d - The cue comparisons without clauses about deviations, variance or levels\n"]
    L.append("The reading guideline classes as unspecified a mention whose level or "
             "variability is described without a direction, and in the validation sample the "
             "author labelled every clause about deviations, variance or a level relative to "
             "normal that way. The extractor counts such a clause with the sign of the nearby "
             "word. This recomputes the comparisons of Section 4.4 with and without the "
             f"clauses that use that wording (pattern `{OFF_TREND.pattern}`), with the "
             "cluster bootstrap and permutation of p3_1_cluster_inference.py and its seed. "
             "The pattern is a lexical proxy for the guideline's rule, fixed after the "
             "reading; the analysis is a sensitivity check, not a replacement for the figures "
             "of record.\n")
    n_flag, n_unsp = sample_check()
    L.append(f"In the validation sample {n_flag} pairs have a matched clause that uses that "
             f"wording, and the author labelled {n_unsp} of them unspecified.\n")
    cued = c.sensor.isin(CUED)
    zs = c["mode"] == "zero_shot"
    L.append("| Claims | Total | With that wording | Share |")
    L.append("|---|---|---|---|")
    for name, m in [("named sensors", cued), ("s4 and s7", ~cued),
                    ("zero-shot runs", zs), ("few-shot runs", ~zs)]:
        k = int((m & c.off_trend).sum())
        L.append(f"| {name} | {int(m.sum()):,} | {k} | {100 * k / m.sum():.1f}% |")

    out = [dict(set="Validation sample", contrast="pairs whose clause uses that wording",
                estimate=n_flag),
           dict(set="Validation sample", contrast="of those, labelled unspecified by the author",
                estimate=n_unsp)]
    contrasts(c, "All claims (the figures of record)", L, out)
    contrasts(c[~c.off_trend].reset_index(drop=True),
              "Without clauses about deviations, variance or levels", L, out)

    pd.DataFrame(out).to_csv(P1 / "p3_3_extractor_sensitivity.csv", index=False)
    (P1 / "p3_3_extractor_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_3_extractor_sensitivity.md'}")


if __name__ == "__main__":
    main()
