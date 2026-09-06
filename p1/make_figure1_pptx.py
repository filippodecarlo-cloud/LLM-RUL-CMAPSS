"""
Figure 1 (methodology + evaluation protocol) as an editable PowerPoint.

Figure 1 is a diagram, not a chart, so Excel is the wrong tool. This writes
native PowerPoint shapes: every box, arrow and label can be moved, recoloured
or retyped in PowerPoint itself. To place it in Word, select all shapes, group
them, copy, and use Paste Special > Picture (Enhanced Metafile) so it stays
vector.

Output: figure1_methodology.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Pt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = EXP / "figure1_methodology.pptx"

INK = RGBColor(0x1F, 0x3B, 0x57)
MID = RGBColor(0x4E, 0x8F, 0xA8)
LIGHT = RGBColor(0x9F, 0xC5, 0xD4)
ACCENT = RGBColor(0xC2, 0x55, 0x3F)
GREY = RGBColor(0x8C, 0x8C, 0x8C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xF2, 0xF6, 0xF8)


def box(slide, x, y, w, h, text, fill=WHITE, line=INK, bold=False, size=10,
        font_colour=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(1.0)
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Cm(0.15)
    tf.margin_top = tf.margin_bottom = Cm(0.05)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Calibri"
    r.font.color.rgb = font_colour
    return s


def arrow(slide, x1, y1, x2, y2, colour=GREY, width=1.25):
    c = slide.shapes.add_connector(2, Cm(x1), Cm(y1), Cm(x2), Cm(y2))  # straight
    c.line.color.rgb = colour
    c.line.width = Pt(width)
    return c


def label(slide, x, y, w, text, size=9, colour=GREY, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.name = "Calibri"
    r.font.italic = not bold
    r.font.bold = bold
    r.font.color.rgb = colour
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Cm(26)
    prs.slide_height = Cm(15)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank

    # No title here: the caption is set in the manuscript, and one baked
    # into the artwork would appear twice on the page.

    # ---- row 1: data to prediction -------------------------------------
    y = 1.6
    box(slide, 0.6, y, 3.6, 1.6,
        "C-MAPSS\nFD001-FD004", fill=PALE, bold=True)
    box(slide, 4.7, y, 3.6, 1.6,
        "Last n cycles\nn = 5 / 15 / 30\nmin-max scaled", fill=PALE)
    box(slide, 8.8, y, 4.2, 1.6,
        "Prompt template\nzero-shot / few-shot", fill=PALE)
    box(slide, 13.5, y - 0.3, 4.6, 2.2,
        "LLM via Ollama\nllama3.1 8B (Q4, Q8)\nmistral 7B, qwen2.5 7B\ndeepseek-r1 7B", fill=PALE,
        size=9)
    box(slide, 18.6, y, 3.4, 1.6, "Parse\nRUL + reasoning", fill=PALE)

    for x1, x2 in ((4.2, 4.7), (8.3, 8.8), (13.0, 13.5), (18.1, 18.6)):
        arrow(slide, x1, y + 0.8, x2, y + 0.8)

    # ---- split into the two things being evaluated ---------------------
    box(slide, 19.0, y + 2.3, 2.6, 1.0, "RUL number", fill=INK, line=INK,
        font_colour=WHITE, bold=True, size=10)
    box(slide, 18.6, y + 3.7, 2.6, 1.0, "Explanation", fill=ACCENT, line=ACCENT,
        font_colour=WHITE, bold=True, size=10)
    arrow(slide, 19.9, y + 1.6, 19.9, y + 2.3)
    arrow(slide, 19.9, y + 3.3, 19.9, y + 3.7)

    # ---- baselines feed the same comparison ----------------------------
    box(slide, 0.6, y + 2.6, 7.7, 1.3,
        "Baselines on the identical split: constant predictors (train mean, "
        "RMSE- and NASA-optimal), random, Random Forest, XGBoost, LSTM",
        fill=WHITE, line=MID, size=9)
    arrow(slide, 2.4, y + 1.6, 2.4, y + 2.6)

    # ---- the protocol ---------------------------------------------------
    py = 8.3
    box(slide, 0.6, py - 0.9, 21.0, 0.8,
        "EVALUATION PROTOCOL", fill=INK, line=INK, font_colour=WHITE,
        bold=True, size=11, shape=MSO_SHAPE.RECTANGLE)

    w, gap = 5.0, 0.35
    items = [
        ("1. No-skill baselines",
         "Is the model better than\npredicting one constant?", INK),
        ("2. Rank correlation",
         "Spearman rho, bootstrap CI,\nBH correction. Does it order\nthe engines?", INK),
        ("3. Output entropy",
         "Distinct values and modal\nshare. Is it regressing or\nchoosing from a menu?", INK),
        ("4. Explanation faithfulness",
         "Asserted direction vs\nthe window actually shown\nvs the measured direction", ACCENT),
    ]
    for i, (title, body, colour) in enumerate(items):
        x = 0.6 + i * (w + gap)
        box(slide, x, py, w, 0.8, title, fill=colour, line=colour,
            font_colour=WHITE, bold=True, size=10, shape=MSO_SHAPE.RECTANGLE)
        box(slide, x, py + 0.8, w, 2.0, body, fill=WHITE, line=colour, size=9,
            shape=MSO_SHAPE.RECTANGLE)

    # ---- the three controls hanging off part 4 -------------------------
    cy = py + 3.2
    box(slide, 16.85, cy, 4.75, 1.9,
        "Controls\n- cue removed\n- cue inverted\n- sensor trends reversed",
        fill=WHITE, line=ACCENT, size=9)
    arrow(slide, 19.2, py + 2.8, 19.2, cy, colour=ACCENT)

    label(slide, 0.6, cy + 0.2, 16,
          "Parts 1-3 test whether the number carries prognostic information. "
          "Part 4 tests whether the text describes the input it was given.",
          size=10, colour=INK)
    label(slide, 0.6, cy + 1.0, 16,
          "Every arm is paired: same engines, same model, same template - one thing "
          "changes at a time.", size=10, colour=INK)

    prs.save(OUT)
    print(f"[saved] {OUT}")
    print("Editable in PowerPoint. To place in Word: select all, group, copy,")
    print("then Paste Special > Picture (Enhanced Metafile) to keep it vector.")


if __name__ == "__main__":
    main()
