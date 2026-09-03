"""
Single source of truth for every number that goes into v14.

Regenerates `numeri_chiave.md` straight from the result CSVs and trace files.
When a number in the manuscript disagrees with this file, this file is right and
the manuscript is stale - it is derived from the data, not retyped from a report.

Run it again after any new experiment, then re-check the manuscript against it.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"
# In the project tree the scripts live under experiment/, so the report belongs
# one level up. In the standalone repository they sit at the top level, and
# writing to the parent would put the file outside the checkout.
OUT = (EXP.parent if EXP.name == "experiment" else EXP) / "numeri_chiave.md"

L = []


def add(s=""):
    L.append(s)


def main():
    add("# Numeri chiave per v14 — fonte unica di verita\n")
    add(f"**Generato automaticamente** da `experiment/p1/make_key_numbers.py` il "
        f"{date.today().isoformat()}, direttamente dai CSV e dalle tracce.\n")
    add("> Se un numero nel manoscritto non coincide con questo file, **questo file ha "
        "ragione**: e derivato dai dati, non ricopiato da un report. Rilancia lo script dopo "
        "ogni nuovo esperimento.\n")

    # ---------- inference count ----------
    tot = 0
    for pat in ("results/traces_*.json", "results_p0/traces_p0_*.json",
                "results_p1/traces_p*.json"):
        for f in EXP.glob(pat):
            tot += len(json.loads(f.read_text(encoding="utf-8")))
    add(f"\n**Inferenze LLM totali del progetto:** {tot:,}\n")

    # ---------- no-skill ----------
    add("\n## Baseline no-skill (FD001)\n")
    b = pd.read_csv(P0 / "p0_1_noskill_baselines.csv")
    b = b[b.dataset == "FD001"]
    add("| Predittore | RMSE | MAE | NASA |")
    add("|---|---|---|---|")
    for _, r in b.iterrows():
        add(f"| {r.predictor} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.NASA_Score:,.0f} |")

    cfg = pd.read_csv(P0 / "p0_1_llm_configs.csv")
    n_beat = int(cfg.beats_trainmean_RMSE.sum())
    add(f"\n- Configurazioni LLM che battono la costante media-train: **{n_beat}/{len(cfg)}**")
    add(f"- Miglior RMSE LLM (FD001): **{cfg[cfg.dataset == 'FD001'].RMSE.min():.2f}**")
    add(f"- Miglior NASA LLM (FD001): **{cfg[cfg.dataset == 'FD001'].NASA_Score.min():,.0f}**")

    # ---------- rank correlation ----------
    add("\n## Correlazione di rango\n")
    rk = pd.read_csv(P0 / "p0_1b_rank_significance.csv")
    add(f"- Test nella famiglia: {len(rk)}")
    add(f"- Nominalmente significativi (p<0,05): **{int((rk.spearman_p < 0.05).sum())}**")
    add(f"- Sopravvivono a BH-FDR 5%: **{int(rk.sig_fdr_5pct.sum())}**")
    add(f"- Sopravvivono a Bonferroni 5%: **{int(rk.sig_bonf_5pct.sum())}**")
    add(f"- Massimo |rho| su configurazioni LLM: **{rk.spearman_rho.abs().max():.3f}** "
        f"(varianza dei ranghi spiegata: {100 * rk.spearman_rho.abs().max() ** 2:.1f}%)")

    # ---------- baselines ----------
    add("\n## Baseline supervisionate (P2.1, dopo riparazione)\n")
    s = pd.read_csv(P1 / "p2_1_baseline_summary.csv")
    add("| Dataset | Modello | RMSE | MAE | NASA | rho | distinti | costante? |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        rho = "n/a" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.3f}"
        add(f"| {r.dataset} | {r.model} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.NASA_Score:,.0f} | "
            f"{rho} | {int(r.distinct_values)} | "
            f"{'SI' if r.degenerate_constant else 'no'} |")

    # ---------- model sweep ----------
    add("\n## Collasso per modello (FD001, zero-shot, n=30)\n")
    add("| Modello | motori | RMSE | distinti | modale (quota) | entropia (bit) |")
    add("|---|---|---|---|---|---|")
    runs = [("llama3.1:8b Q4_K_M", EXP / "results" / "traces_FD001_zero_shot_k0_n30.json")]
    for f in sorted(P1.glob("traces_p1_FD001_zs_n30_*.json")):
        runs.append((f.stem.replace("traces_p1_FD001_zs_n30_", ""), f))
    for label, f in runs:
        if not f.exists():
            continue
        tr = json.loads(f.read_text(encoding="utf-8"))
        p = np.array([t["pred_rul"] for t in tr if t.get("pred_rul") is not None], float)
        if not len(p):
            continue
        v, c = np.unique(p, return_counts=True)
        q = c / c.sum()
        y = np.array([t["true_rul"] for t in tr if t.get("pred_rul") is not None], float)
        flag = "" if len(p) >= 100 else f" *(campione ridotto, n={len(p)})*"
        add(f"| {label}{flag} | {len(p)} | {np.sqrt(((p - y) ** 2).mean()):.2f} | {len(v)} | "
            f"{int(v[c.argmax()])} ({100 * c.max() / len(p):.0f}%) | "
            f"{-(q * np.log2(q)).sum():.2f} |")

    # ---------- multi-condition ----------
    p22 = P1 / "p2_2_summary.csv"
    if p22.exists():
        add("\n## Condizioni operative multiple (P2.2)\n")
        m = pd.read_csv(p22)
        add("| Dataset | Modello | n | RMSE | rho | distinti | modale (quota) |")
        add("|---|---|---|---|---|---|---|")
        for _, r in m.iterrows():
            rho = "n/a" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.3f}"
            add(f"| {r.dataset} | {r.model} | {int(r.n)} | {r.RMSE:.2f} | {rho} | "
                f"{int(r.distinct_values)} | {int(r.modal_value)} "
                f"({r.modal_share_pct:.0f}%) |")

    # ---------- faithfulness ----------
    add("\n## Fedelta delle spiegazioni (P0.2a / P0.3)\n")
    cl = pd.read_csv(P0 / "p0_2a_claims.csv")
    d = cl[cl.claimed.isin(["increase", "decrease"])]
    add(f"- Opportunita sensore-traccia: **{len(cl):,}**; affermazioni direzionali: **{len(d):,}**")
    add(f"- Accordo con la direzione canonica: **{100 * (d.claimed == d.expected_textbook).mean():.1f}%**")
    add(f"- Fedelta alla finestra mostrata: **{100 * (d.claimed == d.actual_in_window).mean():.1f}%**")
    add(f"- Tasso di base (asserire sempre il canonico): "
        f"**{100 * (cl.expected_textbook == cl.actual_in_window).mean():.1f}%**")
    cued = d[d.cued]
    unc = d[~d.cued]
    add(f"- Sensori citati nel prompt: {100 * (cued.claimed == cued.expected_textbook).mean():.1f}% "
        f"vs non citati: {100 * (unc.claimed == unc.expected_textbook).mean():.1f}%")

    arms = P0 / "p0_23_arm_summary.csv"
    if arms.exists():
        a = pd.read_csv(arms)
        add("\n### Bracci di controllo\n")
        add("| Braccio | RMSE | distinti | modale (quota) |")
        add("|---|---|---|---|")
        for _, r in a.iterrows():
            add(f"| {r.arm} | {r.RMSE:.2f} | {int(r.distinct_values)} | "
                f"{int(r.modal_value)} ({r.modal_share_pct:.0f}%) |")

    # ---------- anchoring ----------
    an = P1 / "p1_34_analysis.csv"
    if an.exists():
        import importlib.util
        import config as cfg2
        spec = importlib.util.spec_from_file_location("fs", HERE / "p1_34_fewshot_sampling.py")
        fs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fs)
        cfg2.N_CYCLES_IN_PROMPT = fs.N_CYCLES
        train_df, _, _ = fs.ex.load_cmapss(fs.DATASET)
        a = pd.read_csv(an)
        a = a[(a.n >= 100) & (a.condition != "published")]
        xs, ys = [], []
        for _, r in a.iterrows():
            cond, seed = r.run.rsplit("_seed", 1)
            _, exs = fs.pick_examples(train_df, cond, int(seed))
            xs.append(float(np.mean([e["true_rul"] for e in exs])))
            ys.append(float(r.pred_mean))
        pr = pearsonr(xs, ys)
        sl, ic = np.polyfit(xs, ys, 1)
        add("\n## Ancoraggio (P1.3)\n")
        add(f"- Set di esempi testati: **{len(xs)}**")
        add(f"- Pearson r fra RUL medio degli esempi e media predetta: "
            f"**{pr.statistic:+.3f}** (p={pr.pvalue:.4f})")
        add(f"- **Varianza dell'output spiegata dagli esempi: {100 * pr.statistic ** 2:.0f}%**")
        add(f"- Retta: media_pred = {ic:.2f} + {sl:.3f} x media_esempi")
        add(f"- Media predetta minima/massima: {min(ys):.1f} / {max(ys):.1f} "
            "(media reale del test: 74,5)")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n... [{len(L)} righe]\n[saved] {OUT}")


if __name__ == "__main__":
    main()
