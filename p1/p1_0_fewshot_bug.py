"""
P1.0 - The few-shot examples in the published experiment are all RUL = 0,
and k does not do what the paper says it does.

Reviewer R1.10 asked whether the released code matches the few-shot procedure
described in the manuscript. It does not. This script demonstrates it two ways.

STATIC. `run_llm` builds its example pool as

    pool = train_df.groupby("engine_id").last()

which is the LAST cycle of each training engine. By construction the piecewise
RUL target is 0 at the last cycle, so every engine in the pool has RUL = 0.
The stratification that follows,

    pool["bin"] = pd.cut(pool["RUL"], bins=[0,40,80,125], ...)

therefore puts all 100 engines in bin 0, and

    per_bin = max(1, k // 3)
    for b in range(3): chosen.extend(pool[pool.bin==b].engine_id[:per_bin])

draws per_bin engines from bin 0 and none from bins 1 and 2. The number of
examples actually placed in the prompt is per_bin, not k:

    k=3  -> 1 example    k=5  -> 1 example    k=10 -> 3 examples

all labelled "True RUL: 0" and all described as "advanced wear".

EMPIRICAL. If k=3 and k=5 build the same prompt, their predictions must agree
like two replicates of one prompt at T=0.1, and must agree with k=10 (a
genuinely different prompt) considerably less. Both hold. Agreement is measured
as excess over the chance agreement implied by each run's own output
distribution, because collapsed outputs agree often by accident.

Outputs (results_p1/):
  p1_0_fewshot_bug_report.md
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import experiment as ex  # noqa: E402

OUT = EXP / "results_p1"
OUT.mkdir(exist_ok=True)


def preds(path):
    return {t["engine_id"]: t["pred_rul"]
            for t in json.loads(Path(path).read_text(encoding="utf-8"))}


def agreement(a, b):
    """(observed agreement %, chance agreement % from marginals, excess)."""
    ids = sorted(set(a) & set(b))
    va = np.array([a[i] for i in ids])
    vb = np.array([b[i] for i in ids])
    obs = float((va == vb).mean() * 100)
    pa = {v: (va == v).mean() for v in set(va)}
    pb = {v: (vb == v).mean() for v in set(vb)}
    ch = float(sum(pa.get(v, 0) * pb.get(v, 0) for v in set(va) | set(vb)) * 100)
    return obs, ch, obs - ch


def main():
    L = ["# P1.0 - The published few-shot procedure does not match the paper\n"]

    # ---------- static analysis ----------
    train_df, _, _ = ex.load_cmapss("FD001")
    pool = train_df.groupby("engine_id").last().reset_index()
    pool["bin"] = pd.cut(pool["RUL"], bins=[0, 40, 80, 125],
                         labels=[0, 1, 2], include_lowest=True).astype(int)

    L.append("\n## 1. Static analysis of `experiment.run_llm`\n")
    L.append(f"- The example pool is the last cycle of each training engine: "
             f"**{len(pool)} engines, RUL min {pool.RUL.min():.0f}, "
             f"max {pool.RUL.max():.0f}** - every one of them is "
             f"{'zero' if pool.RUL.max() == 0 else 'nonzero'}, because the piecewise "
             "target is 0 at failure.")
    L.append(f"- Stratifying that pool by RUL bin gives "
             f"**{pool['bin'].value_counts().to_dict()}**: bins 1 and 2 are empty.")
    L.append("- So the loop over the three bins can only draw from bin 0, and the prompt "
             "receives `per_bin = max(1, k//3)` examples rather than `k`:\n")
    L.append("| k requested | per_bin | examples actually in the prompt | engines | RUL shown |")
    L.append("|---|---|---|---|---|")
    for k in (3, 5, 10):
        per_bin = max(1, k // 3)
        chosen = []
        for b in range(3):
            chosen.extend(pool[pool["bin"] == b]["engine_id"].tolist()[:per_bin])
        chosen = chosen[:k]
        ruls = [int(train_df[train_df.engine_id == e]["RUL"].iloc[-1]) for e in chosen]
        L.append(f"| {k} | {per_bin} | **{len(chosen)}** | {chosen} | {ruls} |")
    L.append("\nEvery example is labelled `True RUL: 0` and, since 0 <= 40, described by "
             "`_build_example_block` as *\"advanced wear; significant deviation in "
             "efficiency sensors\"*. The model is shown only end-of-life engines and asked "
             "to extrapolate to healthy ones.\n")

    # ---------- empirical check ----------
    L.append("\n## 2. Empirical check on the published traces\n")
    reps = [EXP / "results" / f"traces_FD001_zero_shot_k0_n30{s}.json"
            for s in ("", "_run2", "_run3", "_run4")]
    reps += [EXP / "results_p0" / "traces_p0_FD001_zs_n30_repl.json"]
    reps = [r for r in reps if r.exists()]
    ref = [agreement(preds(a), preds(b)) for a, b in itertools.combinations(reps, 2)]
    ref_excess = float(np.mean([e for _, _, e in ref]))
    L.append(f"Reference for *same prompt, T=0.1*: {len(reps)} replicate zero-shot n=30 runs, "
             f"{len(ref)} pairs. Mean observed agreement "
             f"{np.mean([o for o, _, _ in ref]):.1f}%, chance "
             f"{np.mean([c for _, c, _ in ref]):.1f}%, **excess "
             f"{ref_excess:+.1f} points**.\n")

    L.append("| Dataset | n | k3 vs k5 (same prompt predicted) | k3 vs k10 (different prompt) |")
    L.append("|---|---|---|---|")
    e35, e310 = [], []
    for ds in ("FD001", "FD003"):
        for suf in ("", "_n15", "_n30"):
            f = EXP / "results"
            a, b, c = (preds(f / f"traces_{ds}_few_shot_k{k}{suf}.json") for k in (3, 5, 10))
            o1, c1, x1 = agreement(a, b)
            o2, c2, x2 = agreement(a, c)
            e35.append(x1)
            e310.append(x2)
            L.append(f"| {ds} | {suf[2:] or '5'} | {o1:.0f}% vs {c1:.0f}% chance "
                     f"(**{x1:+.1f}**) | {o2:.0f}% vs {c2:.0f}% chance (**{x2:+.1f}**) |")

    L.append(f"\nMean excess agreement: **k3-k5 {np.mean(e35):+.1f} points**, in line with the "
             f"{ref_excess:+.1f} of true replicates of one prompt; **k3-k10 "
             f"{np.mean(e310):+.1f} points**, clearly lower. The pattern matches the static "
             "analysis: k=3 and k=5 are the same single-example prompt, k=10 is a different "
             "three-example one.\n")

    # ---------- consequences ----------
    L.append("\n## 3. What this invalidates\n")
    L.append("- **The k ablation is not an ablation over k.** Every 'k=3' and 'k=5' result in "
             "the paper comes from one example; 'k=10' from three. Any statement of the form "
             "'performance improves with more examples' is unsupported - two of the three "
             "levels are the same condition.")
    L.append("- **The stratification described in the methods never happened.** All examples "
             "sit in one RUL bin, and it is the bin at failure.")
    L.append("- **This is a plausible mechanism for the low anchors.** The model is shown "
             "engines with RUL 0 described as badly worn, then asked about a test set whose "
             f"mean RUL is 74.5. Few-shot predictions collapse onto 23, 45 and 95 - all far "
             "below the mean. This is now a testable hypothesis rather than speculation "
             "(reviewer R4.7), and P1.3 tests it by supplying genuinely stratified examples.")
    L.append("- **Reviewer R1.10 was right to ask.** The released code and the described "
             "procedure diverge, and the divergence has to be disclosed in v14 whatever else "
             "changes.")
    L.append("\n> Note: the categorical collapse itself does **not** depend on this defect. "
             "Zero-shot prompts contain no examples at all and collapse harder (P0.4), and "
             "the collapse reproduces on mistral, qwen2.5 and llama-q8_0 (P1.2). What the "
             "defect invalidates is specifically the k ablation and the stratification claim.")

    (OUT / "p1_0_fewshot_bug_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p1_0_fewshot_bug_report.md'}")


if __name__ == "__main__":
    main()
