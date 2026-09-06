"""
Export the figures to PNG by driving Excel and PowerPoint.

The charts live as native Excel objects in figures_v14.xlsx and as PowerPoint
shapes in figure1_methodology.pptx and graphical_abstract.pptx, so they stay
editable. This renders each of them to PNG for insertion into the manuscript.

If a figure needs fixing later, edit the Excel or PowerPoint file, run this
again, and replace that one image in Word by hand.

Output: figures_png/fig1.png ... fig8.png, graphical_abstract.png

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
            # enlarged to reach print density. 1600 px across a 17 cm figure is
            # about 240 dpi; the font sizes below compensate for the scaling.
            ch.Width = 1600
            ch.Height = 960
            try:
                ch.Chart.ChartArea.Format.TextFrame2.TextRange.Font.Size = 20
            except Exception:
                pass
            dest = OUT / f"fig{n}.png"
            ch.Chart.Export(str(dest), "PNG")
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
