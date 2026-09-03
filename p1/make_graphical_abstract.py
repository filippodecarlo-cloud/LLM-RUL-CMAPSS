"""
Graphical abstract: the trend-reversal result, as editable PowerPoint shapes.

Elsevier asks for a graphical abstract at least 1328 x 531 px. The slide here is
26 x 10.4 cm, the same 2.5:1 proportion, so exporting at 300 dpi gives roughly
3070 x 1228 px with room to spare.

Everything is a native shape: move it, recolour it, retype it in PowerPoint.
To submit, export the slide as PNG or TIFF at 300 dpi.

Output: graphical_abstract.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "graphical_abstract.pptx"

INK = RGBColor(0x1F, 0x3B, 0x57)
MID = RGBColor(0x4E, 0x8F, 0xA8)
ACCENT = RGBColor(0xC2, 0x55, 0x3F)
GREY = RGBColor(0x8C, 0x8C, 0x8C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xF2, 0xF6, 0xF8)


def text(slide, x, y, w, h, s, size=11, bold=False, colour=INK,
         align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = s
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = "Calibri"
    r.font.color.rgb = colour
    return tb


def box(slide, x, y, w, h, fill=WHITE, line=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(1.25)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def sparkline(slide, x, y, w, h, rising=True, colour=MID):
    """A small freeform trend line, drawn as a rotated straight connector."""
    y1, y2 = (y + h, y) if rising else (y, y + h)
    c = slide.shapes.add_connector(2, Cm(x), Cm(y1), Cm(x + w), Cm(y2))
    c.line.color.rgb = colour
    c.line.width = Pt(2.5)
    return c


def main():
    prs = Presentation()
    prs.slide_width = Cm(26)
    prs.slide_height = Cm(10.4)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    text(s, 0.6, 0.35, 25, 1,
         "Do the explanations describe the data?", size=17, bold=True)
    text(s, 0.6, 1.25, 25, 0.8,
         "The same 100 turbofan engines, the same model, the same prompt. "
         "Only the direction of every sensor trend changes.",
         size=10.5, colour=GREY, italic=True)

    # ---- left panel: real data ----
    box(s, 0.6, 2.5, 8.4, 6.9, fill=PALE, line=MID)
    text(s, 1.0, 2.75, 7.6, 0.7, "ORIGINAL DATA", size=10, bold=True, colour=MID,
         align=PP_ALIGN.CENTER)
    for i, (lab, rising) in enumerate([("s11", True), ("s9", False), ("s14", False)]):
        yy = 3.7 + i * 1.0
        text(s, 1.1, yy - 0.12, 1.2, 0.6, lab, size=10, bold=True)
        sparkline(s, 2.4, yy, 2.6, 0.55, rising=rising)
    box(s, 1.1, 6.9, 7.4, 2.0, fill=WHITE, line=MID)
    text(s, 1.35, 7.1, 6.9, 1.7,
         "“s11 shows increasing values, indicating advanced wear; "
         "s9 and s14 are trending downwards”",
         size=9.5, italic=True, colour=INK)

    # ---- arrow ----
    a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Cm(9.3), Cm(5.4), Cm(1.6), Cm(1.1))
    a.fill.solid()
    a.fill.fore_color.rgb = ACCENT
    a.line.fill.background()
    a.shadow.inherit = False
    text(s, 8.9, 4.5, 2.4, 0.6, "reversed\nin time", size=8.5, colour=ACCENT,
         align=PP_ALIGN.CENTER, bold=True)

    # ---- right panel: reversed data ----
    box(s, 11.2, 2.5, 8.4, 6.9, fill=PALE, line=ACCENT)
    text(s, 11.6, 2.75, 7.6, 0.7, "EVERY TREND FLIPPED", size=10, bold=True,
         colour=ACCENT, align=PP_ALIGN.CENTER)
    for i, (lab, rising) in enumerate([("s11", False), ("s9", True), ("s14", True)]):
        yy = 3.7 + i * 1.0
        text(s, 11.7, yy - 0.12, 1.2, 0.6, lab, size=10, bold=True)
        sparkline(s, 13.0, yy, 2.6, 0.55, rising=rising, colour=ACCENT)
    box(s, 11.7, 6.9, 7.4, 2.0, fill=WHITE, line=ACCENT)
    text(s, 11.95, 7.1, 6.9, 1.7,
         "“s11 shows increasing values, indicating advanced wear; "
         "s9 and s14 are trending downwards”",
         size=9.5, italic=True, colour=INK)
    text(s, 11.7, 8.85, 7.4, 0.6, "unchanged", size=9, bold=True, colour=ACCENT,
         align=PP_ALIGN.CENTER)

    # ---- the number ----
    box(s, 20.2, 2.5, 5.2, 6.9, fill=INK, line=INK)
    text(s, 20.4, 3.3, 4.8, 1.6, "2.4%", size=44, bold=True, colour=WHITE,
         align=PP_ALIGN.CENTER)
    text(s, 20.5, 5.3, 4.6, 3.6,
         "of the model's stated sensor directions changed when every trend in "
         "the data was reversed.\n\n253 paired cases.",
         size=10, colour=WHITE, align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(f"[saved] {OUT}")
    print("Editable in PowerPoint. Export the slide as PNG or TIFF at 300 dpi")
    print("for submission (gives about 3070 x 1228 px, above the Elsevier minimum).")


if __name__ == "__main__":
    main()
