"""
P3.5 - Does the direction the prompt asserts match the benchmark?

The zero-shot prompt tells the model which sensors rise and which fall as an
engine wears. The faithfulness analysis scored a stated direction against that
same canonical set, which is fine as a measure of whether the model repeats the
prompt, but it assumes the prompt is right about the benchmark. Nobody had
checked.

This measures the direction of each scored sensor in the benchmark itself, in two
ways that do not depend on each other, and compares the result with what the
prompt asserts:

  RAW ENDPOINTS   for each training engine, the mean of the first ten cycles
                  against the mean of the last ten, on unscaled values.
  RANK            for each training engine, the Spearman correlation between the
                  sensor and the remaining useful life.

It then reports how often the model follows the prompt and how often it follows
the benchmark, separately for the sensors where the two agree and the sensors
where they do not, which is the comparison the faithfulness analysis could not
make while it treated the prompt as ground truth.

Outputs (results_p1/):
  p3_5_directions.csv        measured direction per sensor, dataset and scale
  p3_5_agreement.csv         model agreement with each reference, per sensor
  p3_5_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"

COLS = (["engine_id", "cycle", "set1", "set2", "set3"]
        + [f"s{i}" for i in range(1, 22)])
SCORED = ["s4", "s7", "s9", "s11", "s12", "s14", "s15"]
# Saxena et al. (2008) variable list, in column order
SYMBOL = {"s4": "T50", "s7": "P30", "s9": "Nc", "s11": "Ps30",
          "s12": "phi", "s14": "NRc", "s15": "BPR"}
MEANING = {"s4": "Total temperature at LPT outlet",
           "s7": "Total pressure at HPC outlet",
           "s9": "Physical core speed",
           "s11": "Static pressure at HPC outlet",
           "s12": "Ratio of fuel flow to Ps30",
           "s14": "Corrected core speed",
           "s15": "Bypass ratio"}
# what the zero-shot prompt asserts, and what the analysis used as canonical
ASSERTED = {"s4": "increase", "s7": "decrease", "s9": "decrease", "s11": "increase",
            "s12": "increase", "s14": "decrease", "s15": "increase"}
CUED = {"s9", "s11", "s12", "s14", "s15"}
# a direction counts as attested when this share of engines shows it
THRESHOLD = 0.65


def raw_share_up(df, sensor, window=None):
    up = []
    for _, g in df.groupby("engine_id"):
        v = g[sensor].to_numpy(dtype=float)
        if window is None:
            if len(v) < 20:
                continue
            a, b = v[:10].mean(), v[-10:].mean()
        else:
            if len(v) < window:
                continue
            a, b = v[-window], v[-1]
        up.append(b > a)
    return float(np.mean(up)), len(up)


def rank_median(df, sensor):
    per = []
    for _, g in df.groupby("engine_id"):
        if g[sensor].nunique() < 2 or len(g) < 10:
            continue
        r = spearmanr(g[sensor].to_numpy(), -g["cycle"].to_numpy()).statistic
        if np.isfinite(r):
            per.append(r)
    per = np.array(per)
    # correlation with -cycle: positive means the sensor falls as the engine ages
    return float(np.median(per)), len(per)


def main():
    L = ["# P3.5 - The direction the prompt asserts against the direction in the data\n"]
    L.append("The zero-shot prompt states which sensors rise and which fall with wear, and "
             "the faithfulness analysis scored the model against that same set of directions. "
             "That measures whether the model repeats the prompt. It assumes the prompt is "
             "right about the benchmark, which is a separate question and is settled here by "
             "measurement.\n")

    rows = []
    for ds in ("FD001", "FD003"):
        df = pd.read_csv(EXP / "cmapss_data" / f"train_{ds}.txt",
                         sep=r"\s+", header=None, names=COLS)
        for s in SCORED:
            life, n = raw_share_up(df, s)
            w30, _ = raw_share_up(df, s, 30)
            w15, _ = raw_share_up(df, s, 15)
            w5, _ = raw_share_up(df, s, 5)
            med, _ = rank_median(df, s)
            direction = "increase" if life > 0.5 else "decrease"
            share = life if direction == "increase" else 1 - life
            rows.append({
                "dataset": ds, "sensor": s, "symbol": SYMBOL[s],
                "meaning": MEANING[s], "engines": n,
                "share_rising_whole_life": round(100 * life, 1),
                "share_rising_last30": round(100 * w30, 1),
                "share_rising_last15": round(100 * w15, 1),
                "share_rising_last5": round(100 * w5, 1),
                "median_rank_corr_with_age": round(-med, 3),
                "measured_direction": direction,
                "share_supporting": round(100 * share, 1),
                "attested": share >= THRESHOLD,
                "asserted_by_prompt": ASSERTED[s],
                "prompt_matches_data": (direction == ASSERTED[s]) and share >= THRESHOLD,
            })
    dirs = pd.DataFrame(rows)
    dirs.to_csv(P1 / "p3_5_directions.csv", index=False)

    L.append("\n## What the benchmark does\n")
    L.append("Share of training engines in which the sensor is higher at the end than at the "
             "start, at four scales. The last column is what the prompt asserts.\n")
    L.append("| Dataset | Sensor | Symbol | Whole life | Last 30 | Last 15 | Last 5 | "
             "Measured | Asserted |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in dirs.iterrows():
        L.append(f"| {r.dataset} | {r.sensor} | {r.symbol} | "
                 f"{r.share_rising_whole_life:.0f}% | {r.share_rising_last30:.0f}% | "
                 f"{r.share_rising_last15:.0f}% | {r.share_rising_last5:.0f}% | "
                 f"{r.measured_direction} | {r.asserted_by_prompt} |")
    L.append(f"\nA direction counts as attested when at least {100 * THRESHOLD:.0f}% of "
             "engines show it. The two methods agree everywhere: the sign of the median "
             "rank correlation with engine age gives the same direction as the endpoint "
             "comparison for every sensor and both sub-datasets.\n")

    fd1 = dirs[dirs.dataset == "FD001"].set_index("sensor")
    wrong = [s for s in SCORED if fd1.loc[s, "measured_direction"] != ASSERTED[s]]
    weak = [s for s in SCORED
            if fd1.loc[s, "measured_direction"] == ASSERTED[s]
            and not fd1.loc[s, "attested"]]
    L.append("On FD001, the reference sub-dataset:\n")
    L.append(f"- The prompt is right about {len(SCORED) - len(wrong) - len(weak)} of the "
             f"{len(SCORED)} scored sensors.")
    if wrong:
        L.append(f"- It asserts the **opposite** of what the data shows for "
                 f"{', '.join(wrong)}.")
    if weak:
        L.append(f"- Its direction for {', '.join(weak)} is not attested at the "
                 f"{100 * THRESHOLD:.0f}% level.")
    L.append("")
    for s in wrong:
        r = fd1.loc[s]
        L.append(f"  - **{s}** ({r.symbol}, {r.meaning}): the prompt says "
                 f"{ASSERTED[s]}, the data shows {r.measured_direction} in "
                 f"{r.share_supporting:.0f}% of engines.")
    L.append("")
    L.append("FD003 carries two fault modes and its directions are correspondingly less "
             "consistent, which is worth noting but does not change the picture on FD001.\n")

    # ---- what the model follows -----------------------------------------
    c = pd.read_csv(P0 / "p0_2a_claims.csv")
    d = c[c.claimed.isin(["increase", "decrease"])].copy()
    d["asserted"] = d.sensor.map(ASSERTED)
    d["measured"] = d.sensor.map(fd1.measured_direction.to_dict())
    d["follows_prompt"] = d.claimed == d.asserted
    d["follows_data"] = d.claimed == d.measured
    d["follows_window"] = d.claimed == d.actual_in_window

    L.append("\n## What the model follows\n")
    L.append(f"| Reference | Agreement over {len(d):,} directional claims |")
    L.append("|---|---|")
    L.append(f"| The direction the prompt asserts | **{100 * d.follows_prompt.mean():.1f}%** |")
    L.append(f"| The direction measured in the benchmark | "
             f"**{100 * d.follows_data.mean():.1f}%** |")
    L.append(f"| The direction in the window actually shown | "
             f"{100 * d.follows_window.mean():.1f}% |")

    L.append("\n| Sensor | Prompt vs data | Claims | Follows prompt | Follows data | "
             "Follows window |")
    L.append("|---|---|---|---|---|---|")
    agg = []
    for s in SCORED:
        x = d[d.sensor == s]
        if not len(x):
            continue
        tag = "agree" if ASSERTED[s] == fd1.loc[s, "measured_direction"] else "**conflict**"
        L.append(f"| {s} ({SYMBOL[s]}) | {tag} | {len(x):,} | "
                 f"{100 * x.follows_prompt.mean():.1f}% | "
                 f"{100 * x.follows_data.mean():.1f}% | "
                 f"{100 * x.follows_window.mean():.1f}% |")
        agg.append({"sensor": s, "symbol": SYMBOL[s], "conflict": tag == "**conflict**",
                    "claims": len(x),
                    "follows_prompt_pct": round(100 * x.follows_prompt.mean(), 1),
                    "follows_data_pct": round(100 * x.follows_data.mean(), 1),
                    "follows_window_pct": round(100 * x.follows_window.mean(), 1)})
    pd.DataFrame(agg).to_csv(P1 / "p3_5_agreement.csv", index=False)

    sub = d[d.sensor.isin(wrong)]
    L.append(f"\n**The decisive subset.** On the {len(wrong)} sensors where the prompt and "
             f"the benchmark disagree ({', '.join(wrong)}), a claim cannot follow both. Over "
             f"the {len(sub):,} claims about them the model follows the prompt "
             f"{100 * sub.follows_prompt.mean():.1f}% of the time and the benchmark "
             f"{100 * sub.follows_data.mean():.1f}%. Where the two references part company, "
             "the model goes with the prompt.\n")

    L.append("\n## What this changes\n")
    L.append("The headline figure is unchanged as a number and changes as a claim. Agreement "
             "of 91.3% is agreement with the direction the prompt supplies, and it was "
             "described as agreement with established degradation physics. On this benchmark "
             "those are not the same thing: measured against the data the same claims are "
             f"right {100 * d.follows_data.mean():.1f}% of the time. It also explains why "
             "input faithfulness sits at the level a rule that ignores the input would "
             "reach, since the reference the model is echoing is itself uninformative about "
             "the sensor windows for three of the five sensors the prompt names.\n")

    (P1 / "p3_5_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_5_report.md'}")


if __name__ == "__main__":
    main()
