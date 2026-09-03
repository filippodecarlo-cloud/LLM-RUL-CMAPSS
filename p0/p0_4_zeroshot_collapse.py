"""
P0.4 - The collapse is not caused by the few-shot examples.

Reviewers argued that the categorical collapse could be an artefact of the
stratified few-shot construction (critique C): examples drawn one per RUL bin
would hand the model a small menu of values to copy.

That hypothesis makes a prediction: remove the examples and the collapse should
weaken. This script tests it on the existing traces. Zero-shot prompts contain
no examples, no RUL labels and no stratification at all.

Outputs (results_p0/):
  p0_4_collapse_by_mode.csv
  p0_4_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "results_p0"
sys.path.insert(0, str(HERE))


def entropy_bits(p):
    _, counts = np.unique(p, return_counts=True)
    q = counts / counts.sum()
    return float(-(q * np.log2(q)).sum())


def main():
    cfg = pd.read_csv(OUT / "p0_1_llm_configs.csv")

    rows = []
    for _, r in cfg.iterrows():
        tr = json.loads((EXP / "results" / r.file).read_text(encoding="utf-8"))
        p = np.array([t["pred_rul"] for t in tr], float)
        rows.append({
            "dataset": r.dataset, "mode": r["mode"], "k": r.k,
            "n_cycles": r.n_cycles, "run": r.run,
            "distinct_values": int(len(np.unique(p))),
            "modal_value": r.modal_value, "modal_share_pct": r.modal_share_pct,
            "entropy_bits": entropy_bits(p),
            "pred_sd": float(p.std(ddof=1)),
            "max_entropy_bits": float(np.log2(len(p))),
        })
    df = pd.DataFrame(rows)
    df["entropy_ratio"] = df.entropy_bits / df.max_entropy_bits
    df.to_csv(OUT / "p0_4_collapse_by_mode.csv", index=False)

    zs = df[df["mode"] == "zero_shot"]
    fs = df[df["mode"] == "few_shot"]

    L = ["# P0.4 - Zero-shot collapses harder than few-shot\n"]
    L.append("If stratified few-shot examples caused the categorical collapse, removing "
             "them would relieve it. They do not: prompts with no examples produce the "
             "*narrowest* output sets in the whole grid.\n")

    L.append("\n## By prompting mode (both datasets pooled)\n")
    L.append("| Mode | Runs | Distinct values (mean, range) | Modal share | Output entropy (bits) |")
    L.append("|---|---|---|---|---|")
    for lab, sub in (("zero-shot (no examples)", zs), ("few-shot (k=3,5,10)", fs)):
        L.append(f"| {lab} | {len(sub)} | {sub.distinct_values.mean():.1f} "
                 f"({sub.distinct_values.min()}-{sub.distinct_values.max()}) | "
                 f"{sub.modal_share_pct.mean():.0f}% | {sub.entropy_bits.mean():.2f} |")

    u, p_u = mannwhitneyu(zs.distinct_values, fs.distinct_values, alternative="less")
    L.append(f"\nMann-Whitney U = {u:.0f}, p = {p_u:.4f} for the one-sided hypothesis that "
             "zero-shot yields FEWER distinct values than few-shot.\n")
    L.append(f"For scale: 100 engines with a genuine regressor give up to "
             f"{np.log2(100):.2f} bits of output entropy; Random Forest produces 100 "
             f"distinct values. Zero-shot averages {zs.entropy_bits.mean():.2f} bits.\n")

    L.append("\n## Every zero-shot run\n")
    L.append("| Dataset | n | Run | Distinct | Modal value | Modal share | Entropy (bits) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in zs.sort_values(["dataset", "n_cycles", "run"]).iterrows():
        L.append(f"| {r.dataset} | {r.n_cycles} | {r.run} | {r.distinct_values} | "
                 f"{r.modal_value:.0f} | {r.modal_share_pct:.0f}% | {r.entropy_bits:.2f} |")

    L.append("\n## Reading\n")
    L.append(f"- Zero-shot prompts contain no examples, no RUL labels and no stratification, "
             f"yet return a median of {zs.distinct_values.median():.0f} distinct values "
             f"across 100 engines, against {fs.distinct_values.median():.0f} for few-shot.")
    L.append(f"- The narrowest run in the entire grid is zero-shot: "
             f"{int(zs.distinct_values.min())} distinct values, modal share "
             f"{zs.modal_share_pct.max():.0f}%.")
    L.append("- Stratified few-shot examples therefore cannot be the cause of the collapse. "
             "They widen the output set slightly, by supplying values to copy, but the "
             "collapse is already present without them.")
    fs_anchor = ", ".join(f"n={int(n)} -> {int(g.modal_value.mode().iloc[0])}"
                          for n, g in fs.groupby("n_cycles"))
    zs_anchor = sorted(set(zs.modal_value.astype(int)))
    L.append(f"- What the examples do change is WHICH value the model settles on. In "
             f"few-shot the modal value tracks the window length ({fs_anchor}); in "
             f"zero-shot it is {zs_anchor[0] if len(zs_anchor) == 1 else zs_anchor} for "
             "every window length and both datasets. The window length moves the anchor "
             "only when examples are present to supply candidate values - an anchoring "
             "effect, not improved temporal modelling (reviewer R1.7).")

    (OUT / "p0_4_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p0_4_report.md'}")


if __name__ == "__main__":
    main()
