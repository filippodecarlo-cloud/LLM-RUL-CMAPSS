"""
P3.4 - Where the canonical sensor directions come from.

The faithfulness analysis scores a stated direction against a canonical one, and
the manuscript described those canonical directions as well established without
saying on what evidence. Two things are done here instead of an appeal to
authority.

IDENTITY. Each sensor column is matched to the measurement it carries, using the
variable list of Saxena et al. (2008), which is the primary description of the
C-MAPSS output. The mapping is checked against the data: the columns that are
constant across the benchmark must be exactly the ones that carry demanded and
ambient quantities under that mapping, which is a falsifiable check rather than
an assumption.

DIRECTION. The direction of each sensor under degradation is measured in the
training set instead of being asserted. For every training engine the Spearman
correlation between the sensor value and the remaining useful life is computed,
and the canonical direction is the sign of the median of those per-engine
correlations. A sensor whose value rises as RUL falls is an increasing sensor.
This makes the reference against which explanations are scored an empirical
quantity of the benchmark rather than a claim about turbofan physics.

Output: results_p1/p3_4_sensor_directions.csv and .md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P1 = EXP / "results_p1"
import config as cfg  # noqa: E402
import experiment as ex  # noqa: E402

# Variable list of Saxena et al. (2008), Table 2, in the order the columns appear.
SAXENA = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
          "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
          "PCNfR_dmd", "W31", "W32"]
DESCR = {
    "T2": "Total temperature at fan inlet", "T24": "Total temperature at LPC outlet",
    "T30": "Total temperature at HPC outlet", "T50": "Total temperature at LPT outlet",
    "P2": "Pressure at fan inlet", "P15": "Total pressure in bypass duct",
    "P30": "Total pressure at HPC outlet", "Nf": "Physical fan speed",
    "Nc": "Physical core speed", "epr": "Engine pressure ratio",
    "Ps30": "Static pressure at HPC outlet", "phi": "Ratio of fuel flow to Ps30",
    "NRf": "Corrected fan speed", "NRc": "Corrected core speed", "BPR": "Bypass ratio",
    "farB": "Burner fuel-air ratio", "htBleed": "Bleed enthalpy",
    "Nf_dmd": "Demanded fan speed", "PCNfR_dmd": "Demanded corrected fan speed",
    "W31": "HPT coolant bleed", "W32": "LPT coolant bleed",
}
SCORED = ["s4", "s7", "s9", "s11", "s12", "s14", "s15"]


def main():
    cols = (["engine_id", "cycle", "set1", "set2", "set3"]
            + [f"s{i}" for i in range(1, 22)])
    L = ["# P3.4 - Sensor identities and measured degradation directions\n"]

    # ---- identity check --------------------------------------------------
    raw = pd.read_csv(EXP / "cmapss_data" / "train_FD001.txt",
                      sep=r"\s+", header=None, names=cols)
    nun = raw[[f"s{i}" for i in range(1, 22)]].nunique()
    constant = [c for c in nun.index if nun[c] == 1]
    expected = [f"s{i + 1}" for i, s in enumerate(SAXENA)
                if s in {"T2", "P2", "epr", "farB", "Nf_dmd", "PCNfR_dmd"}]
    L.append("## The mapping from column to measurement\n")
    L.append("Column order follows the variable list of Saxena et al. (2008). The check is "
             "that the columns which carry ambient or demanded quantities, and therefore "
             "cannot vary in a fixed-condition simulation, are exactly the columns that are "
             "constant in the data.\n")
    L.append(f"- Constant in FD001: {', '.join(constant)}")
    L.append(f"- Predicted constant under this mapping: {', '.join(expected)}")
    L.append(f"- **Match: {'yes' if set(constant) == set(expected) else 'NO'}**\n")

    # ---- measured directions --------------------------------------------
    rows = []
    for ds in ("FD001", "FD003"):
        train, _, _ = ex.load_cmapss(ds)
        for s in SCORED:
            per = []
            for _, g in train.groupby("engine_id"):
                if g[s].nunique() < 2 or len(g) < 10:
                    continue
                r = spearmanr(g[s].to_numpy(), g["RUL"].to_numpy()).statistic
                if np.isfinite(r):
                    per.append(r)
            per = np.array(per)
            med = float(np.median(per))
            # a sensor that rises as RUL falls has a negative rho with RUL
            direction = "increase" if med < 0 else "decrease"
            share = float((np.sign(per) == np.sign(med)).mean())
            idx = int(s[1:]) - 1
            rows.append({"sensor": s, "symbol": SAXENA[idx],
                         "measurement": DESCR[SAXENA[idx]], "dataset": ds,
                         "median_rho_with_RUL": round(med, 3),
                         "engines_agreeing_pct": round(100 * share, 1),
                         "n_engines": len(per), "measured_direction": direction})
    df = pd.DataFrame(rows)
    df.to_csv(P1 / "p3_4_sensor_directions.csv", index=False)

    L.append("\n## Direction under degradation, measured on the training set\n")
    L.append("Per training engine, the Spearman correlation between the sensor and the "
             "remaining useful life; the direction is the sign of the median across engines. "
             "A negative correlation with RUL means the sensor rises as the engine wears.\n")
    L.append("| Sensor | Saxena symbol | Measurement | Dataset | Median rho with RUL | "
             "Engines agreeing | Measured direction |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        L.append(f"| {r.sensor} | {r.symbol} | {r.measurement} | {r.dataset} | "
                 f"{r.median_rho_with_RUL:+.3f} | {r.engines_agreeing_pct:.0f}% of "
                 f"{r.n_engines} | {r.measured_direction} |")

    pivot = df.pivot(index="sensor", columns="dataset", values="measured_direction")
    agree = (pivot.FD001 == pivot.FD003)
    L.append(f"\nThe two sub-datasets agree on {int(agree.sum())} of {len(agree)} sensors"
             + ("" if agree.all() else f" (disagreement: {list(agree[~agree].index)})") + ".\n")

    # Same attestation threshold as p3_5, so the two scripts classify the same
    # sensors. p3_5 measures the share of engines from the endpoint comparison,
    # this one from the per-engine rank correlation; at 65% the two agree on
    # every sensor and sub-dataset.
    THRESHOLD = 65
    hard = df[df.engines_agreeing_pct < THRESHOLD]
    if len(hard):
        L.append(f"Not every direction is attested. These show the measured direction in "
                 f"fewer than {THRESHOLD}% of engines, the threshold used in p3_5, so they "
                 "have no consistent direction to be scored against and are excluded from the "
                 "benchmark-agreement figures there:\n")
        for _, r in hard.iterrows():
            L.append(f"- {r.sensor} ({r.symbol}) on {r.dataset}: "
                     f"{r.engines_agreeing_pct:.0f}% of engines, median rho "
                     f"{r.median_rho_with_RUL:+.3f}")
        L.append("")
    near = df[(df.engines_agreeing_pct >= THRESHOLD) & (df.engines_agreeing_pct < 80)]
    if len(near):
        L.append("These clear the threshold without being strongly attested, and are kept:\n")
        for _, r in near.iterrows():
            L.append(f"- {r.sensor} ({r.symbol}) on {r.dataset}: "
                     f"{r.engines_agreeing_pct:.0f}% of engines, median rho "
                     f"{r.median_rho_with_RUL:+.3f}")
        L.append("")

    L.append("\n## Consequence for the faithfulness analysis\n")
    cued = {"s9", "s11", "s12", "s14", "s15"}
    for s in SCORED:
        sub = df[df.sensor == s]
        w = "named in the prompt" if s in cued else "not named in the prompt"
        L.append(f"- **{s}** ({sub.symbol.iloc[0]}, {sub.measurement.iloc[0]}), {w}: "
                 f"measured direction "
                 f"{'/'.join(sorted(set(sub.measured_direction)))}, attested in "
                 f"{sub.engines_agreeing_pct.min():.0f} to "
                 f"{sub.engines_agreeing_pct.max():.0f}% of engines.")
    L.append("\nThe canonical direction used for scoring is the measured one. Where the two "
             "sub-datasets or the engines within one of them disagree, that is stated rather "
             "than smoothed over.\n")

    (P1 / "p3_4_sensor_directions.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_4_sensor_directions.md'}")


if __name__ == "__main__":
    main()
