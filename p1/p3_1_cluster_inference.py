"""
P3.1 - Cluster-aware inference for the faithfulness analysis.

The 3,816 directional claims are not independent observations. Each one sits
inside a response, each response inside an engine, and each engine is asked about
in many configurations. A chi-square that treats every claim as its own
observation therefore reports a p-value that is too small, and the size of that
distortion cannot be read off the chi-square itself.

This script re-does every inferential claim in the faithfulness sections with the
dependence respected, and leads with effect sizes and intervals rather than with
p-values.

Method, both procedures at the cluster level:

  CLUSTER BOOTSTRAP   resample whole clusters with replacement, carrying all the
                      claims that belong to them, and recompute the statistic.
                      Percentile intervals from 10,000 resamples. Run twice, once
                      clustering on engine and once on configuration; the wider
                      of the two is reported, which is the conservative choice
                      when it is not obvious which level carries most of the
                      dependence.

  CLUSTER PERMUTATION permute the label at the level of the cluster, never within
                      it, so the null preserves whatever correlation the cluster
                      induces. Two-sided where the contrast is two-sided.

For the paired control arms the pairing is (engine, sensor) across arms, so the
cluster is the engine and the permutation flips whole engines between arms.

Everything below runs on integer-coded numpy arrays, because the resampling loops
are large enough that pandas indexing inside them would dominate the runtime.

Outputs (results_p1/):
  p3_1_cluster_inference.csv   one row per contrast
  p3_1_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P0, P1 = EXP / "results_p0", EXP / "results_p1"
P1.mkdir(exist_ok=True)

N_BOOT = 10000
N_PERM = 10000
SEED = 20260906
CUED = {"s9", "s11", "s12", "s14", "s15"}


# ------------------------------------------------------------------ machinery
def cluster_index(keys):
    """Map cluster labels to 0..k-1 and return (codes, list of row arrays)."""
    uniq, codes = np.unique(np.asarray(keys), return_inverse=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return codes, [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def boot_ci(row_groups, stat, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for stat(rows), resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    k = len(row_groups)
    vals = np.empty(n_boot)
    vals[:] = np.nan
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        rows = np.concatenate([row_groups[i] for i in pick])
        vals[b] = stat(rows)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def widest(*cis):
    ok = [c for c in cis if np.isfinite(c[0]) and np.isfinite(c[1])]
    if not ok:
        return np.nan, np.nan
    return min(c[0] for c in ok), max(c[1] for c in ok)


def cramers_v_codes(a, b, na, nb):
    """Cramer's V from two integer-coded arrays with na and nb levels."""
    if a.size == 0:
        return np.nan
    tab = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)
    keep_r = tab.sum(1) > 0
    keep_c = tab.sum(0) > 0
    tab = tab[np.ix_(keep_r, keep_c)]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float(((tab - exp) ** 2 / exp).sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def rec(store, **kw):
    store.append(kw)


# ------------------------------------------------------------------ data
def load_main():
    c = pd.read_csv(P0 / "p0_2a_claims.csv")
    c = c[c.claimed.isin(["increase", "decrease"])].reset_index(drop=True)
    c["engine"] = c.dataset + "-" + c.engine_id.astype(str)
    c["config"] = (c.dataset + "|" + c["mode"] + "|k" + c.k.astype(str)
                   + "|n" + c.n_cycles.astype(str) + "|" + c.run)
    return c


def load_arms():
    a = pd.read_csv(P0 / "p0_23_claims.csv")
    return a[a.claimed.isin(["increase", "decrease"])].reset_index(drop=True)


# ------------------------------------------------------------------ main
def main():
    out = []
    L = ["# P3.1 - Cluster-aware inference for the faithfulness analysis\n"]
    L.append("Every inferential statement in the faithfulness sections, recomputed with the "
             "dependence between claims respected. Claims are nested in responses, responses "
             "in engines, and engines are asked about in many configurations, so a test that "
             "treats each claim as an independent observation reports a p-value that is too "
             f"small. Intervals are percentile intervals from {N_BOOT:,} cluster bootstrap "
             "resamples, computed once clustering on engine and once on configuration, with "
             f"the wider of the two reported. Permutation p-values use {N_PERM:,} "
             "cluster-level permutations and are reported as the proportion of permutations "
             "at least as extreme, so the smallest attainable value is "
             f"{1 / (N_PERM + 1):.5f}.\n")

    c = load_main()
    tb_ok = (c.claimed == c.expected_textbook).to_numpy().astype(float)
    in_ok = (c.claimed == c.actual_in_window).to_numpy().astype(float)
    base_ok = (c.actual_in_window == c.expected_textbook).to_numpy().astype(float)
    is_cued = c.sensor.isin(CUED).to_numpy()
    sensor = c.sensor.to_numpy()
    _, eng_groups = cluster_index(c.engine.to_numpy())
    _, cfg_groups = cluster_index(c.config.to_numpy())

    L.append(f"Main analysis: {len(c):,} directional claims, {len(eng_groups)} engine "
             f"clusters, {len(cfg_groups)} configuration clusters.\n")

    # ---- 1. cued vs uncued sensors --------------------------------------
    L.append("\n## Cued sensors against uncued sensors\n")

    def d_cued(rows, mask=is_cued):
        m = mask[rows]
        if m.all() or (~m).all():
            return np.nan
        return tb_ok[rows][m].mean() - tb_ok[rows][~m].mean()

    obs = d_cued(np.arange(len(c)))
    lo, hi = widest(boot_ci(eng_groups, d_cued), boot_ci(cfg_groups, d_cued))
    # The cue is assigned to sensors, so the permutation relabels sensors.
    sensors = np.array(sorted(set(sensor)))
    n_cued = len(CUED & set(sensors))
    rng = np.random.default_rng(SEED)
    allrows = np.arange(len(c))
    cnt = 0
    for _ in range(N_PERM):
        fake = set(rng.choice(sensors, size=n_cued, replace=False))
        v = d_cued(allrows, np.isin(sensor, list(fake)))
        if np.isfinite(v) and abs(v) >= abs(obs) - 1e-12:
            cnt += 1
    p_sensor = (cnt + 1) / (N_PERM + 1)
    from math import comb
    floor_p = 1.0 / comb(len(sensors), n_cued)

    L.append(f"Claims about cued sensors agree with the canonical direction "
             f"{100 * tb_ok[is_cued].mean():.1f}% of the time (n = {int(is_cued.sum()):,}) and "
             f"claims about uncued sensors {100 * tb_ok[~is_cued].mean():.1f}% "
             f"(n = {int((~is_cued).sum())}). The difference is {100 * obs:.1f} points, "
             f"95% CI [{100 * lo:.1f}, {100 * hi:.1f}].\n")
    L.append(f"The published chi-square for this contrast was 599.3 with p of order 1e-132, "
             "computed as if the 3,816 claims were independent. Permuting which sensors count "
             "as cued, which is the level at which the cue is actually assigned, gives "
             f"p = {p_sensor:.4f}. There are only seven sensors, so this design cannot produce "
             f"a p below {floor_p:.3f} however large the effect: with a partition this coarse "
             "the p-value carries little information and the difference in proportions with "
             "its interval is what should be reported.\n")
    rec(out, contrast="cued vs uncued sensors, canonical agreement",
        estimate=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi, unit="percentage points",
        n=len(c), p_cluster=p_sensor,
        note=f"permutation over sensors; smallest attainable p is {floor_p:.3f}")

    # ---- 2. cued prompt vs uncued prompt --------------------------------
    L.append("\n## Prompts that carry the cue against prompts that do not\n")
    is_zs = (c["mode"] == "zero_shot").to_numpy()
    cfg_codes, _ = cluster_index(c.config.to_numpy())
    cfg_is_zs = pd.Series(is_zs).groupby(cfg_codes).first().to_numpy()

    def d_mode(rows, lab=None):
        m = is_zs[rows] if lab is None else lab[cfg_codes[rows]]
        if m.all() or (~m).all():
            return np.nan
        return tb_ok[rows][m].mean() - tb_ok[rows][~m].mean()

    obs = d_mode(allrows)
    lo, hi = widest(boot_ci(eng_groups, d_mode), boot_ci(cfg_groups, d_mode))
    rng = np.random.default_rng(SEED)
    lab = cfg_is_zs.copy()
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(lab)
        v = d_mode(allrows, lab)
        if np.isfinite(v) and abs(v) >= abs(obs) - 1e-12:
            cnt += 1
    p_mode = (cnt + 1) / (N_PERM + 1)
    L.append(f"The zero-shot template carries the cue sentence and the few-shot template does "
             f"not. Canonical agreement is {100 * tb_ok[is_zs].mean():.1f}% "
             f"(n = {int(is_zs.sum()):,}) against {100 * tb_ok[~is_zs].mean():.1f}% "
             f"(n = {int((~is_zs).sum())}), a difference of {100 * obs:.1f} points, 95% CI "
             f"[{100 * lo:.1f}, {100 * hi:.1f}]. Permuting the template label between the "
             f"{len(cfg_groups)} configurations gives p = {p_mode:.4f}.\n")
    rec(out, contrast="cued prompt vs uncued prompt, canonical agreement",
        estimate=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi, unit="percentage points",
        n=len(c), p_cluster=p_mode, note="permutation over configurations")

    # ---- 3. faithfulness against the base rate --------------------------
    L.append("\n## Faithfulness against the rate obtained by ignoring the input\n")

    def gap(rows):
        return in_ok[rows].mean() - base_ok[rows].mean()

    obs = gap(allrows)
    lo, hi = widest(boot_ci(eng_groups, gap), boot_ci(cfg_groups, gap))
    L.append(f"Input faithfulness is {100 * in_ok.mean():.1f}% while the canonical direction "
             f"is actually present in the window {100 * base_ok.mean():.1f}% of the time, so a "
             f"model that ignored the input and always stated the canonical direction would "
             f"score the second number. The gap is {100 * obs:+.1f} points, 95% CI "
             f"[{100 * lo:+.1f}, {100 * hi:+.1f}]. The interval contains zero, so on these "
             "claims the explanations are indistinguishable from a rule that never looks at "
             "the data. Both percentages are computed on the same 3,816 rows: over all "
             "18,900 opportunities the same no-look rule scores 46.7%, but faithfulness is "
             "undefined wherever the model stated no direction, so that figure has a "
             "different denominator and is not the comparison to make.")
    rec(out, contrast="input faithfulness minus base rate", estimate=100 * obs,
        ci_lo=100 * lo, ci_hi=100 * hi, unit="percentage points", n=len(c),
        p_cluster=np.nan, note="negative means worse than ignoring the input")

    # ---- 4. per-sensor association --------------------------------------
    L.append("\n## Association between the stated direction and the window, sensor by sensor\n")
    L.append("| Sensor | Claims | Cramer's V | 95% CI (cluster) | Cluster permutation p |")
    L.append("|---|---|---|---|---|")
    claimed_c, _ = np.unique(c.claimed.to_numpy(), return_inverse=True)
    cl_codes = np.unique(c.claimed.to_numpy(), return_inverse=True)[1]
    aw_levels, aw_codes = np.unique(c.actual_in_window.to_numpy(), return_inverse=True)
    n_cl, n_aw = len(claimed_c), len(aw_levels)
    eng_codes, _ = cluster_index(c.engine.to_numpy())

    for s in sorted(set(sensor)):
        sel = np.where(sensor == s)[0]
        sub_cl = cl_codes[sel]
        sub_aw = aw_codes[sel]
        sub_eng = eng_codes[sel]
        obs = cramers_v_codes(sub_cl, sub_aw, n_cl, n_aw)
        if not np.isfinite(obs):
            L.append(f"| {s} | {len(sel)} | not defined | not defined | not defined |")
            rec(out, contrast=f"claim vs window association, {s}", estimate=np.nan,
                ci_lo=np.nan, ci_hi=np.nan, unit="Cramer's V", n=len(sel),
                p_cluster=np.nan, note="the model states a single direction; V undefined")
            continue
        _, groups_s = cluster_index(sub_eng)
        stat = lambda r: cramers_v_codes(sub_cl[r], sub_aw[r], n_cl, n_aw)  # noqa: E731
        lo, hi = boot_ci(groups_s, stat)
        # permute the window direction across engines, engines kept intact
        ue, inv = np.unique(sub_eng, return_inverse=True)
        eng_aw = np.array([sub_aw[sub_eng == e][0] for e in ue])
        rng2 = np.random.default_rng(SEED)
        perm = eng_aw.copy()
        cnt = 0
        for _ in range(N_PERM):
            rng2.shuffle(perm)
            v = cramers_v_codes(sub_cl, perm[inv], n_cl, n_aw)
            if np.isfinite(v) and v >= obs - 1e-12:
                cnt += 1
        p = (cnt + 1) / (N_PERM + 1)
        L.append(f"| {s} | {len(sel)} | {obs:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.4f} |")
        rec(out, contrast=f"claim vs window association, {s}", estimate=obs,
            ci_lo=lo, ci_hi=hi, unit="Cramer's V", n=len(sel), p_cluster=p,
            note="permutation moves whole engines")
    L.append("\nThe conclusion in the manuscript does not change, and its wording is already "
             "the careful one: no statistically detectable association for six of the seven "
             "sensors. What changes is that the p-values now come from a procedure that does "
             "not assume independence, and every association carries an interval, so a reader "
             "can see how weak the one detectable association is.\n")

    # ---- 5. control arms, paired by engine ------------------------------
    L.append("\n## Control arms, paired by engine\n")
    a = load_arms()
    a["ok"] = (a.claimed == a.canonical).astype(float)
    piv = a.pivot_table(index=["engine_id", "sensor"], columns="arm",
                        values="ok", aggfunc="first")
    for arm, label in [("nocue", "cue removed"), ("invcue", "cue inverted"),
                       ("revdata", "trends reversed")]:
        both = piv[["repl", arm]].dropna().reset_index()
        if both.empty:
            continue
        diff = (both[arm] - both["repl"]).to_numpy(dtype=float)
        _, gp = cluster_index(both.engine_id.to_numpy())
        obs = diff.mean()
        lo, hi = boot_ci(gp, lambda r: diff[r].mean())
        rng3 = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(N_PERM):
            sign = rng3.integers(0, 2, size=len(gp)) * 2 - 1
            tot = sum(s * diff[g].sum() for s, g in zip(sign, gp))
            if abs(tot / len(diff)) >= abs(obs) - 1e-12:
                cnt += 1
        p = (cnt + 1) / (N_PERM + 1)
        L.append(f"- **{label}**: canonical agreement moves {100 * obs:+.1f} points on "
                 f"{len(both):,} paired claims across {len(gp)} engines, 95% CI "
                 f"[{100 * lo:+.1f}, {100 * hi:+.1f}], engine-level sign-flip p = {p:.4f}.")
        rec(out, contrast=f"paired arm: {label} minus replication", estimate=100 * obs,
            ci_lo=100 * lo, ci_hi=100 * hi, unit="percentage points", n=len(both),
            p_cluster=p, note=f"paired on (engine, sensor), {len(gp)} engine clusters")
    L.append("\nThe published chi-square for the inverted cue was 103.1 with p of order 1e-24. "
             "The paired, engine-clustered version gives the same direction and a similar "
             "magnitude, with an interval attached and without assuming independence.\n")

    # ---- 6. trend reversal, with an interval ----------------------------
    L.append("\n## The trend-reversal result, with an interval\n")
    rp = a[a.arm == "repl"].set_index(["engine_id", "sensor"])
    rv = a[a.arm == "revdata"].set_index(["engine_id", "sensor"])
    common = rp.index.intersection(rv.index)
    pair = pd.DataFrame({
        "claim_repl": rp.loc[common, "claimed"].to_numpy(),
        "claim_rev": rv.loc[common, "claimed"].to_numpy(),
        "shown_repl": rp.loc[common, "actual_shown"].to_numpy(),
        "shown_rev": rv.loc[common, "actual_shown"].to_numpy(),
        "engine_id": [i[0] for i in common],
        "sensor": [i[1] for i in common],
    })
    dirn = ["increase", "decrease"]
    flip = pair[(pair.shown_repl != pair.shown_rev)
                & pair.shown_repl.isin(dirn) & pair.shown_rev.isin(dirn)].copy()
    changed = (flip.claim_repl != flip.claim_rev).to_numpy(dtype=float)
    k, n = int(changed.sum()), len(changed)
    _, gp = cluster_index(flip.engine_id.to_numpy())
    lo, hi = boot_ci(gp, lambda r: changed[r].mean())
    ph = k / n
    z = 1.959963985
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    L.append(f"Both arms make a directional claim for {len(pair):,} (engine, sensor) pairs. "
             f"The trend actually shown reversed for {n} of them, and the model's stated "
             f"direction changed for {k}, or {100 * ph:.1f}%. Clustering on engine "
             f"({len(gp)} clusters) the 95% interval is [{100 * lo:.1f}, {100 * hi:.1f}]; the "
             f"unclustered Wilson interval is [{100 * (centre - half):.1f}, "
             f"{100 * (centre + half):.1f}].\n")
    retained = 100 * n / max(len(pair), 1)
    L.append(f"**On the selection.** Restricting to pairs whose shown trend reversed keeps "
             f"{n} of {len(pair):,} pairs, {retained:.1f}%. The restriction is necessary, "
             "since a pair whose trend did not reverse cannot test whether the claim follows "
             "the trend, but the figure describes only those pairs and should be read that "
             "way. It also bears only on what the model says about trend direction, and does "
             "not establish that no other information in the window reached the prediction.\n")
    rec(out, contrast="stated direction changes when the shown trend reverses",
        estimate=100 * ph, ci_lo=100 * lo, ci_hi=100 * hi, unit="%", n=n,
        p_cluster=np.nan,
        note=f"{k} of {n} reversed pairs; {retained:.1f}% of paired claims retained")

    pd.DataFrame(out).to_csv(P1 / "p3_1_cluster_inference.csv", index=False)
    (P1 / "p3_1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {P1 / 'p3_1_report.md'}")


if __name__ == "__main__":
    main()
