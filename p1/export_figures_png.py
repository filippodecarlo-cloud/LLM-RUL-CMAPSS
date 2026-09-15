"""
Export the figures to PNG by driving Excel and PowerPoint.

The charts live as native Excel objects in figures_v14.xlsx and as PowerPoint
shapes in figure1_methodology.pptx and graphical_abstract.pptx, so they stay
editable. This renders each of them to PNG for insertion into the manuscript.

If a figure needs fixing later, edit the Excel or PowerPoint file, run this
again, and replace that one image in Word by hand.

Output: figures_png/fig1.png ... fig6.png, fig8.png, fig9.png, graphical_abstract.png.
fig7.png (the s11 trend-reversal worked example) is not built here - it is a
standalone PNG from p1/p3_9_trend_reversal_figure.py.

    python p1/export_figures_png.py
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "figures_png"
XLSX = EXP / "figures_v14.xlsx"
PPTX_FIG1 = EXP / "figure1_methodology.pptx"
PPTX_GA = EXP / "graphical_abstract.pptx"


# Excel exports at the object's size in points times the display scaling, so the
# same workbook gave 3200 px on one run and 7533 px on another. Every figure is
# resampled to one width afterwards, which makes the output independent of the
# machine. 2400 px across 15.5 cm is about 390 dpi.
EXPORT_PT = 1600
TARGET_W = 2400


def normalise(path):
    """Resample an exported PNG to TARGET_W, so the pipeline is deterministic."""
    from PIL import Image
    im = Image.open(path)
    if im.width == TARGET_W:
        return
    if im.width < TARGET_W:
        print(f"    {path.name}: exported at {im.width} px, below the {TARGET_W} px "
              "target; left as it is rather than upsampled")
        return
    h = round(im.height * TARGET_W / im.width)
    im.resize((TARGET_W, h), Image.LANCZOS).save(path)


def _get(fn, default=None):
    """COM raises for anything a chart does not have; absence is not an error."""
    try:
        return fn()
    except Exception:
        return default


def _set(fn):
    try:
        fn()
    except Exception:
        pass


def read_sizes(chart):
    """Every point-valued property, read before the object is enlarged."""
    s = {"axes": [], "series": []}
    s["title"] = _get(lambda: chart.ChartTitle.Font.Size) if _get(
        lambda: chart.HasTitle, False) else None
    s["legend"] = _get(lambda: chart.Legend.Font.Size) if _get(
        lambda: chart.HasLegend, False) else None
    for i in (1, 2):
        ax = _get(lambda: chart.Axes(i))
        if ax is None:
            s["axes"].append(None)
            continue
        s["axes"].append({
            "ticks": _get(lambda: ax.TickLabels.Font.Size),
            "title": _get(lambda: ax.AxisTitle.Font.Size) if _get(
                lambda: ax.HasTitle, False) else None,
            "line": (_get(lambda: ax.Format.Line.Weight)
                     if _get(lambda: ax.Format.Line.Visible, 0) == -1 else None),
            "grid": _get(lambda: ax.MajorGridlines.Format.Line.Weight) if _get(
                lambda: ax.HasMajorGridlines, False) else None,
        })
    n = _get(lambda: chart.SeriesCollection().Count, 0)
    for i in range(1, n + 1):
        ser = chart.SeriesCollection(i)
        s["series"].append({
            "marker": _get(lambda: ser.MarkerSize) if _get(
                lambda: ser.MarkerStyle, -4142) != -4142 else None,
            # writing a weight onto a hidden line turns it on, so a series
            # drawn as markers only must be left alone
            "line": (_get(lambda: ser.Format.Line.Weight)
                     if _get(lambda: ser.Format.Line.Visible, 0) == -1 else None),
            "labels": _get(lambda: ser.Points(1).DataLabel.Font.Size),
            "points": _get(lambda: ser.Points().Count, 0),
            # a trendline is not a series and keeps its own weight
            "trend": _get(lambda: ser.Trendlines(1).Format.Line.Weight),
        })
    return s


def apply_sizes(chart, s, f):
    """Multiply what was read by the same factor the geometry grew by."""
    def pts(v, lo=1.0, hi=409.0):
        return max(lo, min(hi, v * f))

    if s["title"]:
        _set(lambda: setattr(chart.ChartTitle.Font, "Size", pts(s["title"])))
    if s["legend"]:
        _set(lambda: setattr(chart.Legend.Font, "Size", pts(s["legend"])))
    for i, a in zip((1, 2), s["axes"]):
        if not a:
            continue
        ax = _get(lambda: chart.Axes(i))
        if ax is None:
            continue
        if a["ticks"]:
            _set(lambda: setattr(ax.TickLabels.Font, "Size", pts(a["ticks"])))
        if a["title"]:
            _set(lambda: setattr(ax.AxisTitle.Font, "Size", pts(a["title"])))
        if a["line"]:
            _set(lambda: setattr(ax.Format.Line, "Weight", pts(a["line"], 0.25, 20)))
        if a["grid"]:
            _set(lambda: setattr(ax.MajorGridlines.Format.Line, "Weight",
                                 pts(a["grid"], 0.25, 20)))
    for i, d in enumerate(s["series"], start=1):
        ser = _get(lambda: chart.SeriesCollection(i))
        if ser is None:
            continue
        if d["marker"]:
            # Excel caps MarkerSize at 72 points
            _set(lambda: setattr(ser, "MarkerSize", int(round(pts(d["marker"], 2, 72)))))
        if d["line"]:
            _set(lambda: setattr(ser.Format.Line, "Weight", pts(d["line"], 0.25, 20)))
        if d["trend"]:
            _set(lambda: setattr(ser.Trendlines(1).Format.Line, "Weight",
                                 pts(d["trend"], 0.25, 20)))
        if d["labels"]:
            size = pts(d["labels"])
            # labels carry their own formatting, point by point
            for j in range(1, int(d["points"]) + 1):
                _set(lambda: setattr(ser.Points(j).DataLabel.Font, "Size", size))


def export_excel_charts():
    import win32com.client as win32
    made = []
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    # Chart axis labels are rendered with the machine's decimal separator, so an
    # Italian Windows writes "0,2" on an English figure. Overriding it here fixes
    # the exported PNG without touching the author's Excel settings.
    sep_saved = excel.UseSystemSeparators
    try:
        excel.UseSystemSeparators = False
        excel.DecimalSeparator = "."
        excel.ThousandsSeparator = ","
    except Exception:
        pass
    try:
        wb = excel.Workbooks.Open(str(XLSX), ReadOnly=True)
        for ws in wb.Worksheets:
            name = ws.Name
            if not name.startswith("FIG_"):
                continue
            n = name.replace("FIG_", "")
            if ws.ChartObjects().Count == 0:
                print(f"  {name}: no chart object, skipped")
                continue
            ch = ws.ChartObjects(1)
            # Chart.Export writes one pixel per point, so the object has to be
            # enlarged to reach print density. Font auto-scaling is turned on
            # first, which makes the enlargement a uniform zoom: the point sizes
            # set in the workbook keep their proportion to the figure, and the
            # normalisation below returns them to the page at those sizes.
            # There is deliberately no font override here. One used to force
            # every string to 20 pt, which put about 5 pt on the printed page and
            # discarded every size chosen in make_figures_xlsx.py.
            aspect = float(ch.Width) / float(ch.Height)
            factor = EXPORT_PT / float(ch.Width)
            sizes = read_sizes(ch.Chart)
            ch.Width = EXPORT_PT
            ch.Height = EXPORT_PT / aspect
            apply_sizes(ch.Chart, sizes, factor)
            dest = OUT / f"fig{n}.png"
            ch.Chart.Export(str(dest), "PNG")
            normalise(dest)
            made.append(dest)
            print(f"  {name} -> {dest.name}")
        wb.Close(SaveChanges=False)
    finally:
        try:
            excel.UseSystemSeparators = sep_saved
        except Exception:
            pass
        excel.Quit()
        time.sleep(1)
    return made


def export_pptx(path, dest):
    import win32com.client as win32
    ppt = win32.DispatchEx("PowerPoint.Application")
    try:
        pres = ppt.Presentations.Open(str(path), WithWindow=False)
        slide = pres.Slides(1)
        # 300 dpi on the slide's own geometry
        w = int(pres.PageSetup.SlideWidth / 72 * 300)
        h = int(pres.PageSetup.SlideHeight / 72 * 300)
        slide.Export(str(dest), "PNG", w, h)
        pres.Close()
        print(f"  {path.name} -> {dest.name} ({w}x{h} px)")
    finally:
        ppt.Quit()
        time.sleep(1)
    return dest


def main():
    OUT.mkdir(exist_ok=True)
    print("Excel charts:")
    made = export_excel_charts()
    print("PowerPoint:")
    if PPTX_FIG1.exists():
        export_pptx(PPTX_FIG1, OUT / "fig1.png")
    if PPTX_GA.exists():
        export_pptx(PPTX_GA, OUT / "graphical_abstract.png")

    print(f"\n{len(list(OUT.glob('*.png')))} PNG in {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name:26s} {p.stat().st_size / 1024:7.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
