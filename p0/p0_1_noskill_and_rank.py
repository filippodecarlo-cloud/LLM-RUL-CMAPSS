"""
P0.1 - No-skill baselines + rank-correlation audit for ALL LLM configurations.

Answers reviewer critiques A (missing no-skill baseline) and R1.7 (ablations move
the anchor, they do not create prognostic capability).

Pure post-hoc computation on the existing trace JSONs - no LLM inference.

Outputs (in results_p0/):
  p0_1_noskill_baselines.csv   constant / trivial predictors, per dataset
  p0_1_llm_configs.csv         one row per trace file: errors, rank corr, collapse stats
  p0_1_report.md               human-readable summary
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DATA = EXP / "cmapss_data"
TRACES = EXP / "results"
OUT = EXP / "results_p0"
OUT.mkdir(exist_ok=True)

RUL_CAP = 125
ALL_COLS = (["engine_id", "cycle", "setting1", "setting2", "setting3"] +
            [f"s{i}" for i in range(1, 22)])


# -- metrics ---------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def nasa_score(y, p):
    d = np.asarray(p, float) - np.asarray(y, float)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


# -- data ------------------------------------------------------------------
def load_truth(dataset):
    """Return (all train RUL labels, test true RUL vector)."""
    train = pd.read_csv(DATA / f"train_{dataset}.txt", sep=r"\s+", header=None,
                        names=ALL_COLS, engine="python").dropna(axis=1)
    rul = pd.read_csv(DATA / f"RUL_{dataset}.txt", sep=r"\s+", header=None,
                      names=["RUL"], engine="python").dropna(axis=1)

    maxc = train.groupby("engine_id")["cycle"].max().rename("max_cycle")
    train = train.join(maxc, on="engine_id")
    train["RUL"] = (train["max_cycle"] - train["cycle"]).clip(upper=RUL_CAP)

    y_test = rul["RUL"].clip(upper=RUL_CAP).values.astype(float)
    return train["RUL"].values.astype(float), y_test


def best_constant(y, metric):
    """Grid-search the constant predictor that optimises `metric` over 0..125."""
    grid = np.arange(0, RUL_CAP + 1, dtype=float)
    vals = [metric(y, np.full_like(y, c)) for c in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), float(vals[i])


# -- trace parsing ---------------------------------------------------------
def parse_tag(path):
    """traces_FD001_few_shot_k10_n30.json -> dict of config fields."""
    stem = path.stem.replace("traces_", "")
    ds = stem[:5]
    rest = stem[6:]
    m = re.match(r"(zero_shot|few_shot)_k(\d+)(?:_n(\d+))?(?:_(run\d+))?$", rest)
    if not m:
        raise ValueError(f"unparsed trace name: {path.name}")
    mode, k, n, run = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    return {"dataset": ds, "mode": mode, "k": k,
            "n_cycles": int(n) if n else 5,
            "run": run or "run1", "file": path.name}


def main():
    truth = {ds: load_truth(ds) for ds in ("FD001", "FD003")}

    # -- no-skill / trivial baselines --------------------------------------
    rows = []
    for ds, (train_rul, y) in truth.items():
        cands = {
            "constant = TRAIN mean RUL": float(train_rul.mean()),
            "constant = TRAIN median RUL": float(np.median(train_rul)),
            "constant = TEST mean RUL (oracle)": float(y.mean()),
            "constant = TEST median RUL (oracle)": float(np.median(y)),
            "constant = 50": 50.0,
            "constant = RUL_CAP/2 (62.5)": 62.5,
        }
        c_rmse, _ = best_constant(y, rmse)
        c_nasa, _ = best_constant(y, nasa_score)
        cands["constant = RMSE-optimal (oracle)"] = c_rmse
        cands["constant = NASA-optimal (oracle)"] = c_nasa

        for name, c in cands.items():
            p = np.full_like(y, c)
            rows.append({"dataset": ds, "predictor": name, "value": round(c, 2),
                         "RMSE": rmse(y, p), "MAE": mae(y, p),
                         "NASA_Score": nasa_score(y, p)})

        # random uniform predictor, 1000 draws, for reference
        rng = np.random.default_rng(42)
        r_rmse, r_mae, r_nasa = [], [], []
        for _ in range(1000):
            p = rng.uniform(0, RUL_CAP, size=len(y))
            r_rmse.append(rmse(y, p))
            r_mae.append(mae(y, p))
            r_nasa.append(nasa_score(y, p))
        rows.append({"dataset": ds, "predictor": "random uniform U(0,125), mean of 1000",
                     "value": np.nan, "RMSE": float(np.mean(r_rmse)),
                     "MAE": float(np.mean(r_mae)), "NASA_Score": float(np.mean(r_nasa))})

    base_df = pd.DataFrame(rows)
    base_df.to_csv(OUT / "p0_1_noskill_baselines.csv", index=False)

    # -- every LLM configuration -------------------------------------------
    cfg_rows = []
    for path in sorted(TRACES.glob("traces_*.json")):
        info = parse_tag(path)
        tr = json.loads(path.read_text(encoding="utf-8"))
        y = np.array([t["true_rul"] for t in tr], float)
        p = np.array([t["pred_rul"] for t in tr], float)

        rho, rho_p = spearmanr(p, y)
        tau, tau_p = kendalltau(p, y)
        vals, counts = np.unique(p, return_counts=True)
        top = counts.max()

        info.update({
            "n_engines": len(y),
            "RMSE": rmse(y, p), "MAE": mae(y, p), "NASA_Score": nasa_score(y, p),
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "kendall_tau": float(tau), "kendall_p": float(tau_p),
            "distinct_values": int(len(vals)),
            "modal_value": float(vals[counts.argmax()]),
            "modal_share_pct": 100.0 * top / len(p),
            "pred_mean": float(p.mean()), "pred_sd": float(p.std(ddof=1)),
            "true_sd": float(y.std(ddof=1)),
            "sd_ratio_pred_over_true": float(p.std(ddof=1) / y.std(ddof=1)),
        })
        # gap vs the honest no-skill reference (train-mean constant)
        train_rul = truth[info["dataset"]][0]
        c = float(train_rul.mean())
        info["RMSE_trainmean_const"] = rmse(y, np.full_like(y, c))
        info["beats_trainmean_RMSE"] = bool(info["RMSE"] < info["RMSE_trainmean_const"])
        cfg_rows.append(info)

    cfg_df = pd.DataFrame(cfg_rows).sort_values(
        ["dataset", "mode", "k", "n_cycles", "run"]).reset_index(drop=True)
    cfg_df.to_csv(OUT / "p0_1_llm_configs.csv", index=False)

    # -- report ------------------------------------------------------------
    L = []
    L.append("# P0.1 - No-skill baselines and rank-correlation audit\n")
    L.append("Post-hoc computation on the existing trace files "
             "(24 unique configurations + 3 zero-shot replicates). No new inference.\n")

    for ds in ("FD001", "FD003"):
        train_rul, y = truth[ds]
        L.append(f"\n## {ds}\n")
        L.append(f"Test RUL (capped at {RUL_CAP}): n={len(y)}, mean={y.mean():.2f}, "
                 f"median={np.median(y):.1f}, sd={y.std(ddof=1):.2f}, "
                 f"range {y.min():.0f}-{y.max():.0f}. "
                 f"Train-label mean RUL = {train_rul.mean():.2f}.\n")

        L.append("\n### No-skill / constant predictors\n")
        sub = base_df[base_df.dataset == ds]
        L.append("| Predictor | c | RMSE | MAE | NASA |")
        L.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            v = "-" if pd.isna(r.value) else f"{r.value:.1f}"
            L.append(f"| {r.predictor} | {v} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.NASA_Score:,.0f} |")

        L.append("\n### LLM configurations\n")
        c = cfg_df[cfg_df.dataset == ds]
        L.append("| Config | RMSE | MAE | NASA | rho | p | distinct | modal (share) | sd_pred/sd_true | beats train-mean? |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in c.iterrows():
            tag = (f"{'ZS' if r['mode'] == 'zero_shot' else 'FS k=' + str(r.k)}, n={r.n_cycles}"
                   + ("" if r.run == "run1" else f" [{r.run}]"))
            L.append(f"| {tag} | {r.RMSE:.2f} | {r.MAE:.2f} | {r.NASA_Score:,.0f} | "
                     f"{r.spearman_rho:+.3f} | {r.spearman_p:.3f} | {r.distinct_values} | "
                     f"{r.modal_value:.0f} ({r.modal_share_pct:.0f}%) | "
                     f"{r.sd_ratio_pred_over_true:.2f} | {'YES' if r.beats_trainmean_RMSE else 'no'} |")

        n_beat = int(c.beats_trainmean_RMSE.sum())
        n_sig = int((c.spearman_p < 0.05).sum())
        L.append(f"\n**Summary {ds}:** {n_beat}/{len(c)} configurations beat the train-mean "
                 f"constant on RMSE; {n_sig}/{len(c)} reach p<0.05 on Spearman rho "
                 f"(max |rho| = {c.spearman_rho.abs().max():.3f}).\n")

    (OUT / "p0_1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT / 'p0_1_noskill_baselines.csv'}")
    print(f"[saved] {OUT / 'p0_1_llm_configs.csv'}")
    print(f"[saved] {OUT / 'p0_1_report.md'}")


if __name__ == "__main__":
    main()
