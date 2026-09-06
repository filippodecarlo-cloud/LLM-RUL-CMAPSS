"""
Editable figures for v14 - native Excel charts, not bitmaps.

Every figure is written as TWO sheets:
  DATI_figN   the numbers, one tidy table, with headers
  FIG_N       a native Excel chart object pointing at those cells

So the chart is a real Excel chart: double-click any element to restyle it,
drag the legend, retype an axis title, change a colour - all in Excel, with no
Python involved. Editing the numbers in DATI_figN redraws the chart. To place a
figure in Word, copy the chart and use Paste Special > Picture (Enhanced
Metafile): it stays vector, so it prints sharp at any size.

Design choices made here on purpose, because they are the details that usually
have to be fixed by hand afterwards:
  * axes start at zero and are shared across panels that must be compared;
  * category order is fixed explicitly, never left to Excel;
  * series names are the ones that should appear in the legend, spelled out;
  * fonts are set to 10 pt so a two-column figure stays legible when shrunk;
  * a print-safe palette that survives greyscale (varying lightness, not hue);
  * gridlines light grey, no chart border, no 3-D, no drop shadows.

Usage:
    python p1/make_figures_xlsx.py                # all available figures
    python p1/make_figures_xlsx.py --only 3       # just one, for a quick look
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

P0 = EXP / "results_p0"
P1 = EXP / "results_p1"
OUTFILE = EXP / "figures_v14.xlsx"

# Print-safe palette: differs in lightness as well as hue, so it survives a
# greyscale printer, which pure-hue palettes do not.
INK = "#1F3B57"      # dark navy
MID = "#4E8FA8"      # mid teal
LIGHT = "#9FC5D4"    # pale blue
ACCENT = "#C2553F"   # brick, for the "what matters" series
GREY = "#8C8C8C"
GRID = "#D9D9D9"

# 11 pt on the axes and 12 pt on titles. A figure reduced to a single column in
# print loses about a third of its linear size, and 10 pt did not survive that.
FONT = {"name": "Calibri", "size": 11}
TITLE_FONT = {"name": "Calibri", "size": 12, "bold": True}
# A thin dark outline separates adjacent bars when the page is printed in
# greyscale, which fill colour alone does not do reliably.
BAR_EDGE = {"color": "#3A3A3A", "width": 0.75}


def style_axes(chart, x_name, y_name, y_max=None, y_min=0):
    chart.set_x_axis({
        "name": x_name, "name_font": FONT, "num_font": FONT,
        "num_format": "General",
        "line": {"color": GREY},
        "major_gridlines": {"visible": False},
    })
    # Integer tick labels: Excel would otherwise inherit the locale's decimal
    # separator and print "100,00" on an English figure.
    y = {"name": y_name, "name_font": FONT, "num_font": FONT,
         "num_format": "0",
         "line": {"color": GREY},
         "major_gridlines": {"visible": True, "line": {"color": GRID, "width": 0.75}},
         "min": y_min}
    if y_max is not None:
        y["max"] = y_max
    chart.set_y_axis(y)
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    chart.set_plotarea({"border": {"none": True}, "fill": {"none": True}})


def write_table(wb, name, df, note=None):
    """Write a tidy table and return (sheet, first_data_row_index)."""
    ws = wb.add_worksheet(name)
    hdr = wb.add_format({"bold": True, "bottom": 1, "font_name": "Calibri",
                         "font_size": 10, "valign": "bottom", "text_wrap": True})
    cell = wb.add_format({"font_name": "Calibri", "font_size": 10})
    num = wb.add_format({"font_name": "Calibri", "font_size": 10, "num_format": "0.00"})

    r0 = 0
    if note:
        ws.write(0, 0, note, wb.add_format({"font_name": "Calibri", "font_size": 9,
                                            "italic": True, "font_color": GREY}))
        r0 = 2
    for j, c in enumerate(df.columns):
        ws.write(r0, j, str(c), hdr)
    for i, (_, row) in enumerate(df.iterrows()):
        for j, v in enumerate(row):
            f = num if isinstance(v, (float, np.floating)) else cell
            ws.write(r0 + 1 + i, j, v.item() if hasattr(v, "item") else v, f)
    ws.set_column(0, 0, 26)
    ws.set_column(1, max(1, len(df.columns) - 1), 14)
    ws.freeze_panes(r0 + 1, 1)
    return ws, r0


def add_chart_sheet(wb, sheet_name, chart, width=1.6, height=1.4):
    ws = wb.add_worksheet(sheet_name)
    ws.hide_gridlines(2)
    ws.insert_chart("B2", chart, {"x_scale": width, "y_scale": height})
    return ws


# ---------------------------------------------------------------- figure 3
def figure_faithfulness(wb):
    """What the stated direction agrees with, per sensor."""
    src = P0 / "p0_2a_claims.csv"
    if not src.exists():
        return False
    cl = pd.read_csv(src)
    d = cl[cl.claimed.isin(["increase", "decrease"])].copy()
    d["prompt_hit"] = d.claimed == d.expected_textbook
    d["input_hit"] = d.claimed == d.actual_in_window
    # direction measured on the FD001 training set, from p1/p3_5_prompt_vs_benchmark.py
    MEASURED = {"s4": "increase", "s7": "decrease", "s9": "increase", "s11": "increase",
                "s12": "decrease", "s14": "increase", "s15": "increase"}
    d["data_hit"] = d.claimed == d.sensor.map(MEASURED)

    rows = []
    for s in ["s4", "s7", "s9", "s11", "s12", "s14", "s15"]:
        sub = d[d.sensor == s]
        if not len(sub):
            continue
        # the base rate has to share its denominator with faithfulness, so it is
        # computed on the directional claims and not on every opportunity
        rows.append({
            "Sensor": f"{s} ({'cued' if s in {'s9','s11','s12','s14','s15'} else 'not cued'})",
            "Agrees with the direction the prompt asserts (%)": 100 * sub.prompt_hit.mean(),
            "Agrees with the direction the data shows (%)": 100 * sub.data_hit.mean(),
            "Agrees with the window supplied (%)": 100 * sub.input_hit.mean(),
            "Base rate, same claims (%)":
                100 * (sub.expected_textbook == sub.actual_in_window).mean(),
            "n claims": int(len(sub)),
        })
    df = pd.DataFrame(rows)

    name = "DATI_fig5"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 5 - what the stated direction agrees with, per sensor. "
                              "Sources: results_p0/p0_2a_report.md and "
                              "results_p1/p3_5_report.md")
    n = len(df)
    first, last = r0 + 1, r0 + n

    chart = wb.add_chart({"type": "column"})
    for col, colour in ((1, INK), (2, ACCENT), (3, LIGHT), (4, GREY)):
        chart.add_series({
            "name": [name, r0, col],
            "categories": [name, first, 0, last, 0],
            "values": [name, first, col, last, col],
            "fill": {"color": colour}, "border": BAR_EDGE,
            "gap": 60,
        })
    chart.set_title({"name": "Directional claims per sensor: what the stated direction agrees with",
                     "name_font": TITLE_FONT})
    style_axes(chart, "Sensor (cued = named in the prompt)", "Percentage of directional claims",
               y_max=100)
    chart.set_legend({"position": "bottom", "font": FONT})
    add_chart_sheet(wb, "FIG_5", chart, 1.9, 1.5)
    return True


# ---------------------------------------------------------------- figure 2
def figure_collapse(wb):
    """Predicted-value distributions: every model, plus a real regressor."""
    runs = [
        ("llama3.1:8b Q4_K_M", EXP / "results" / "traces_FD001_zero_shot_k0_n30.json"),
        ("llama3.1:8b Q8_0", P1 / "traces_p1_FD001_zs_n30_llama3.1_8b-instruct-q8_0.json"),
        ("mistral:7b", P1 / "traces_p1_FD001_zs_n30_mistral_7b.json"),
        ("qwen2.5:7b", P1 / "traces_p1_FD001_zs_n30_qwen2.5_7b.json"),
    ]
    edges = np.arange(0, 131, 10)
    labels = [f"{int(a)}-{int(b - 1)}" for a, b in zip(edges[:-1], edges[1:])]
    data = {"RUL bin": labels}

    for label, path in runs:
        if not path.exists():
            continue
        tr = json.loads(path.read_text(encoding="utf-8"))
        p = np.array([t["pred_rul"] for t in tr if t.get("pred_rul") is not None], float)
        data[label] = np.histogram(p, bins=edges)[0].tolist()

    bp = P1 / "p2_1_baseline_predictions.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        b = b[b.dataset == "FD001"]
        data["XGBoost (reference)"] = np.histogram(b.XGBoost.values, bins=edges)[0].tolist()
        data["True RUL (reference)"] = np.histogram(b.true_rul.values, bins=edges)[0].tolist()

    df = pd.DataFrame(data)
    name = "DATI_fig3"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 3 - how many of the 100 test engines fall in each "
                              "predicted-RUL bin. A regressor spreads out; the LLMs do not.")
    n = len(df)
    first, last = r0 + 1, r0 + n
    palette = [INK, MID, LIGHT, ACCENT, GREY, "#000000"]

    chart = wb.add_chart({"type": "column"})
    for j, col in enumerate(df.columns[1:], start=1):
        chart.add_series({
            "name": [name, r0, j],
            "categories": [name, first, 0, last, 0],
            "values": [name, first, j, last, j],
            "fill": {"color": palette[(j - 1) % len(palette)]}, "border": BAR_EDGE,
            "gap": 40,
        })
    chart.set_title({"name": "Distribution of predicted RUL, FD001 (100 engines)",
                     "name_font": TITLE_FONT})
    style_axes(chart, "Predicted RUL (cycles)", "Number of engines")
    chart.set_legend({"position": "bottom", "font": FONT})
    add_chart_sheet(wb, "FIG_3", chart, 2.1, 1.5)
    return True


# ---------------------------------------------------------------- figure 4
def figure_reversed_trends(wb):
    """The data was turned upside down; the explanations were not."""
    src = P0 / "p0_23_claims.csv"
    if not src.exists():
        return False
    cl = pd.read_csv(src)
    piv = cl[cl.arm.isin(["repl", "revdata"])].pivot_table(
        index=["engine_id", "sensor"], columns="arm",
        values=["claimed", "actual_shown"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna().reset_index()
    both = piv[piv.claimed_repl.isin(["increase", "decrease"]) &
               piv.claimed_revdata.isin(["increase", "decrease"])]

    rows = []
    for s in ["s7", "s9", "s11", "s12", "s14", "s15"]:
        sub = both[both.sensor == s]
        if not len(sub):
            continue
        flipped = sub[sub.actual_shown_repl != sub.actual_shown_revdata]
        changed = flipped[flipped.claimed_repl != flipped.claimed_revdata]
        rows.append({"Sensor": s,
                     "Pairs where the real trend reversed": int(len(flipped)),
                     "Pairs where the model's claim changed": int(len(changed))})
    tot_f = sum(r["Pairs where the real trend reversed"] for r in rows)
    tot_c = sum(r["Pairs where the model's claim changed"] for r in rows)
    rows.append({"Sensor": "ALL", "Pairs where the real trend reversed": tot_f,
                 "Pairs where the model's claim changed": tot_c})
    df = pd.DataFrame(rows)

    name = "DATI_fig6"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 6 - each engine's sensor window was reversed in time, so "
                              "every trend flipped sign. The stated directions did not follow. "
                              "Source: results_p0/p0_23_report.md")
    first, last = r0 + 1, r0 + len(df)
    chart = wb.add_chart({"type": "column"})
    for col, colour, gap in ((1, INK, 60), (2, ACCENT, 60)):
        chart.add_series({
            "name": [name, r0, col],
            "categories": [name, first, 0, last, 0],
            "values": [name, first, col, last, col],
            "fill": {"color": colour}, "border": BAR_EDGE, "gap": gap,
        })
    chart.set_title({"name": f"Paired trend reversals and the claims that followed "
                             f"them ({100 * tot_c / max(tot_f, 1):.1f}%)",
                     "name_font": TITLE_FONT})
    style_axes(chart, "Sensor", "Number of (engine, sensor) pairs")
    chart.set_legend({"position": "bottom", "font": FONT})
    add_chart_sheet(wb, "FIG_6", chart, 1.8, 1.4)
    return True


# ---------------------------------------------------------------- figure 5
def figure_pred_vs_true(wb):
    """Predicted against true RUL: the LLM against models that regress."""
    bp = P1 / "p2_1_baseline_predictions.csv"
    llm_path = EXP / "results" / "traces_FD001_zero_shot_k0_n30.json"
    if not (bp.exists() and llm_path.exists()):
        return False
    b = pd.read_csv(bp)
    b = b[b.dataset == "FD001"].set_index("engine_id")
    tr = json.loads(llm_path.read_text(encoding="utf-8"))
    llm = {t["engine_id"]: t["pred_rul"] for t in tr}

    rows = []
    for eid, r in b.iterrows():
        rows.append({"True RUL": r.true_rul,
                     "llama3.1:8b (zero-shot)": llm.get(eid, np.nan),
                     "XGBoost": r.XGBoost,
                     "LSTM (repaired)": r.LSTM_fixed,
                     "Perfect prediction": r.true_rul})
    df = pd.DataFrame(rows).sort_values("True RUL").reset_index(drop=True)

    name = "DATI_fig2"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 2 - predicted vs true RUL, FD001. Points on the diagonal "
                              "are perfect. The LLM forms horizontal bands: its output does not "
                              "depend on the engine.")
    first, last = r0 + 1, r0 + len(df)

    chart = wb.add_chart({"type": "scatter", "subtype": "markers_only"})
    specs = [(1, ACCENT, "circle", 7), (2, INK, "square", 5), (3, MID, "triangle", 5)]
    for col, colour, sym, size in specs:
        chart.add_series({
            "name": [name, r0, col],
            "categories": [name, first, 0, last, 0],
            "values": [name, first, col, last, col],
            "marker": {"type": sym, "size": size,
                       "fill": {"color": colour}, "border": {"none": True}},
            "line": {"none": True},
        })
    chart.add_series({                       # the y = x reference
        "name": [name, r0, 4],
        "categories": [name, first, 0, last, 0],
        "values": [name, first, 4, last, 4],
        "marker": {"type": "none"},
        "line": {"color": GREY, "width": 1.0, "dash_type": "dash"},
    })
    chart.set_title({"name": "Predicted vs true RUL, FD001 (100 engines)",
                     "name_font": TITLE_FONT})
    style_axes(chart, "True RUL (cycles)", "Predicted RUL (cycles)", y_max=130)
    chart.set_x_axis({"name": "True RUL (cycles)", "name_font": FONT, "num_font": FONT,
                      "num_format": "0",
                      "min": 0, "max": 130, "line": {"color": GREY},
                      "major_gridlines": {"visible": True,
                                          "line": {"color": GRID, "width": 0.75}}})
    chart.set_legend({"position": "bottom", "font": FONT})
    add_chart_sheet(wb, "FIG_2", chart, 1.6, 1.6)
    return True


# ---------------------------------------------------------------- figure 6
def figure_anchoring(wb):
    """Mean prediction follows the mean RUL of the examples in the prompt."""
    src = P1 / "p1_34_analysis.csv"
    if not src.exists():
        return False
    a = pd.read_csv(src)
    a = a[(a.n >= 100) & (a.condition != "published")]

    # Recompute the example RULs from the sampler itself. The summary CSV only
    # holds the last invocation, so reading it would silently drop the `asis`
    # run - which is the one that anchors the low end of this figure.
    import importlib.util
    import config as cfg
    spec = importlib.util.spec_from_file_location(
        "fs", HERE / "p1_34_fewshot_sampling.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    cfg.N_CYCLES_IN_PROMPT = fs.N_CYCLES
    train_df, _, _ = fs.ex.load_cmapss(fs.DATASET)

    ex_mean = {}
    for run in a.run:
        try:
            cond, seed = run.rsplit("_seed", 1)
            _, examples = fs.pick_examples(train_df, cond, int(seed))
            ex_mean[run] = float(np.mean([e["true_rul"] for e in examples]))
        except Exception as exc:
            print(f"    (fig6: could not rebuild examples for {run}: {exc})")

    rows = []
    for _, r in a.iterrows():
        rows.append({"Run": r.run,
                     "Mean RUL of the prompt examples": ex_mean.get(r.run, np.nan),
                     "Mean predicted RUL": r.pred_mean,
                     "Test-set mean RUL": 74.5})
    df = pd.DataFrame(rows).dropna().sort_values("Mean RUL of the prompt examples")

    name = "DATI_fig7"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 7 - the anchor follows the examples. Each point is one "
                              "few-shot example set. Source: results_p1/p1_34_report.md")
    first, last = r0 + 1, r0 + len(df)

    chart = wb.add_chart({"type": "scatter", "subtype": "markers_only"})
    chart.add_series({
        "name": [name, r0, 2],
        "categories": [name, first, 1, last, 1],
        "values": [name, first, 2, last, 2],
        "marker": {"type": "circle", "size": 9, "fill": {"color": ACCENT},
                   "border": {"none": True}},
        "line": {"none": True},
        # the paper leans on r = +0.941; show the fit rather than assert it.
        # the name is set explicitly, or Excel invents one in its own language.
        "trendline": {"type": "linear", "name": "Linear fit",
                      "line": {"color": ACCENT, "width": 1.0, "dash_type": "dash"}},
    })
    chart.add_series({
        "name": [name, r0, 3],
        "categories": [name, first, 1, last, 1],
        "values": [name, first, 3, last, 3],
        "marker": {"type": "none"},
        "line": {"color": GREY, "width": 1.25, "dash_type": "dash"},
    })
    chart.set_title({"name": "Mean predicted RUL versus mean RUL of the few-shot examples (n = 7, r = 0.94)",
                     "name_font": TITLE_FONT})
    style_axes(chart, "Mean RUL of the examples placed in the prompt",
               "Mean predicted RUL (100 engines)", y_max=130)
    chart.set_x_axis({"name": "Mean RUL of the examples placed in the prompt",
                      "name_font": FONT, "num_font": FONT, "num_format": "0",
                      "min": 0, "max": 120, "line": {"color": GREY},
                      "major_gridlines": {"visible": False}})
    chart.set_legend({"position": "bottom", "font": FONT})
    add_chart_sheet(wb, "FIG_7", chart, 1.6, 1.4)
    return True


# ---------------------------------------------------------------- figure 7
def figure_rank_correlation(wb):
    """Rank correlation for every configuration, against the baselines."""
    src = P0 / "p0_1b_rank_significance.csv"
    if not src.exists():
        return False
    c = pd.read_csv(src)
    c = c[c.dataset == "FD001"].copy()
    c["Configuration"] = c.apply(
        lambda r: ("ZS" if r["mode"] == "zero_shot" else f"FS k={int(r.k)}")
                  + f", n={int(r.n_cycles)}"
                  + ("" if r.run == "run1" else f" [{r.run}]"), axis=1)
    c = c.sort_values("spearman_rho")

    rows = []
    for _, r in c.iterrows():
        rows.append({"Configuration": r.Configuration,
                     "Spearman rho": r.spearman_rho,
                     "CI lower error": r.spearman_rho - r.rho_ci_lo,
                     "CI upper error": r.rho_ci_hi - r.spearman_rho,
                     "Random Forest": 0.813,
                     "LSTM (repaired)": 0.889})
    df = pd.DataFrame(rows)

    name = "DATI_fig8"
    ws, r0 = write_table(wb, name, df,
                         note="Figure 8 - rank correlation with 95% bootstrap CI, FD001. The "
                              "two reference lines are models that actually regress.")
    first, last = r0 + 1, r0 + len(df)

    chart = wb.add_chart({"type": "bar"})
    chart.add_series({
        "name": "LLM configurations",
        "categories": [name, first, 0, last, 0],
        "values": [name, first, 1, last, 1],
        "fill": {"color": INK}, "border": {"none": True}, "gap": 40,
        "x_error_bars": {"type": "custom",
                         "plus_values": f"='{name}'!${chr(68)}${first + 1}:${chr(68)}${last + 1}",
                         "minus_values": f"='{name}'!${chr(67)}${first + 1}:${chr(67)}${last + 1}",
                         "line": {"color": GREY, "width": 1.0}},
    })
    chart.set_title({"name": "Rank correlation with the true RUL (FD001)",
                     "name_font": TITLE_FONT})
    chart.set_x_axis({"name": "Spearman rho", "name_font": FONT, "num_font": FONT,
                      "num_format": "[$-409]0.0",
                      "min": -0.4, "max": 1.0, "line": {"color": GREY},
                      "major_gridlines": {"visible": True,
                                          "line": {"color": GRID, "width": 0.75}}})
    chart.set_y_axis({"name": "", "num_font": {"name": "Calibri", "size": 9},
                      "label_position": "low",
                      "line": {"color": GREY}})
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    chart.set_plotarea({"border": {"none": True}, "fill": {"none": True}})
    chart.set_legend({"none": True})
    add_chart_sheet(wb, "FIG_8", chart, 1.7, 1.8)

    ws.write(r0 + len(df) + 3, 0,
             "Reference: Random Forest rho = +0.813, repaired LSTM rho = +0.889 on the same "
             "100 engines. Add them as vertical lines in Excel if you want them on the chart.",
             wb.add_format({"font_name": "Calibri", "font_size": 9, "italic": True,
                            "font_color": GREY}))
    return True


# ---------------------------------------------------------------- figure 8
def figure_multicondition(wb):
    """The collapse across all four CMAPSS subsets, against the baselines."""
    rows = []

    # FD001 and FD003 from the original zero-shot n=30 runs
    for ds in ("FD001", "FD003"):
        p = EXP / "results" / f"traces_{ds}_zero_shot_k0_n30.json"
        if not p.exists():
            continue
        tr = json.loads(p.read_text(encoding="utf-8"))
        v = np.array([t["pred_rul"] for t in tr], float)
        vals, counts = np.unique(v, return_counts=True)
        rows.append({"Dataset": f"{ds}\n1 condition, {len(v)} engines",
                     "Distinct predicted values": int(len(vals)),
                     "Modal share (%)": 100 * counts.max() / len(v),
                     "Engines": len(v)})

    # FD002 / FD004 from P2.2
    for ds in ("FD002", "FD004"):
        p = P1 / f"traces_p2_{ds}_zs_n30.json"
        if not p.exists():
            continue
        tr = json.loads(p.read_text(encoding="utf-8"))
        v = np.array([t["pred_rul"] for t in tr if t.get("pred_rul") is not None], float)
        if not len(v):
            continue
        vals, counts = np.unique(v, return_counts=True)
        rows.append({"Dataset": f"{ds}\n6 conditions, {len(v)} engines",
                     "Distinct predicted values": int(len(vals)),
                     "Modal share (%)": 100 * counts.max() / len(v),
                     "Engines": len(v)})

    if len(rows) < 2:
        return False
    df = pd.DataFrame(rows)

    # Flag any subset still being collected, so a partial run is never read as final.
    expected = {"FD001": 100, "FD003": 100, "FD002": 259, "FD004": 248}
    partial = []
    for _, r in df.iterrows():
        ds = r["Dataset"].split("\n")[0]
        if r["Engines"] < expected.get(ds, 0):
            partial.append(f"{ds}: {int(r['Engines'])}/{expected[ds]}")
    note = ("Figure 4 - the collapse across all four CMAPSS subsets, zero-shot n=30. "
            "A regressor would produce as many distinct values as there are engines.")
    if partial:
        note += "  *** PARTIAL RUN, regenerate when finished: " + "; ".join(partial) + " ***"
        print(f"    (figure 8: partial data - {'; '.join(partial)})")

    name = "DATI_fig4"
    ws, r0 = write_table(wb, name, df, note=note)
    first, last = r0 + 1, r0 + len(df)

    chart = wb.add_chart({"type": "column"})
    chart.add_series({
        "name": [name, r0, 2],
        "categories": [name, first, 0, last, 0],
        "values": [name, first, 2, last, 2],
        "fill": {"color": INK}, "border": BAR_EDGE, "gap": 80,
        # the number of distinct values belongs on the bar, not on a second axis
        "data_labels": {
            "value": True, "font": {"name": "Calibri", "size": 11, "color": INK},
            "position": "outside_end", "num_format": "0",
            "custom": [{"value": f"{int(v)} value{'s' if v != 1 else ''}"}
                       for v in df["Distinct predicted values"]],
        },
    })
    chart.set_title({
        "name": "Share of predictions on a single value, by sub-dataset (zero-shot, n = 30)",
        "name_font": TITLE_FONT})
    style_axes(chart, "Sub-dataset", "Predictions on the modal value (%)", y_max=100)
    chart.set_legend({"none": True})
    add_chart_sheet(wb, "FIG_4", chart, 1.8, 1.4)
    return True


# numbered in order of first citation in the manuscript
BUILDERS = {2: figure_pred_vs_true, 3: figure_collapse, 4: figure_multicondition,
            5: figure_faithfulness, 6: figure_reversed_trends, 7: figure_anchoring,
            8: figure_rank_correlation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(OUTFILE))
    args = ap.parse_args()

    wb = xlsxwriter.Workbook(args.out, {"nan_inf_to_errors": True})
    made = []
    for num, fn in sorted(BUILDERS.items()):
        if args.only and num != args.only:
            continue
        if fn(wb):
            made.append(num)
            print(f"  figure {num}: written")
        else:
            print(f"  figure {num}: SKIPPED (source data not found)")
    wb.close()
    print(f"\n[saved] {args.out}  -  figures {made}")
    print("In Excel: edit anything you like, then copy the chart and use")
    print("Paste Special > Picture (Enhanced Metafile) to place it in Word as vector art.")


if __name__ == "__main__":
    main()
