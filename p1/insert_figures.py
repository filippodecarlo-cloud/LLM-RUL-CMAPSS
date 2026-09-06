"""
Place the figure PNGs into the Word file, each after the paragraph that cites it.

Reads paper_v16_zotero.docx (the version with live citation fields) and writes
paper_v16_submission.docx, the file to submit.

Each figure is inserted centred, with a caption below it in the usual style.
If a figure needs changing later, edit figures_v14.xlsx or the PowerPoint,
re-run export_figures_png.py, and swap that one image in Word by hand.

    python p1/insert_figures.py
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
PNG = EXP / "figures_png"
SRC = ROOT / "paper_v16_zotero.docx"
OUT = ROOT / "paper_v16_submission.docx"

# figure -> (anchor: a distinctive phrase in the paragraph it belongs after, caption)
FIGURES = {
    1: ("Figure 1 shows how the parts fit together",
        "Figure 1. The experimental pipeline and the four-part evaluation protocol. "
        "Parts 1 to 3 test whether the predicted number carries prognostic information; "
        "part 4 tests whether the explanation describes the input it was given."),
    2: ("Figure 2 plots predicted against true RUL",
        "Figure 2. Predicted against true RUL on FD001. Points on the dashed diagonal would be "
        "exact. The prompted model forms horizontal bands: its output does not depend on the "
        "engine."),
    3: ("The anchor is model-specific",
        "Figure 3. Distribution of predicted RUL on FD001 for the four models run on all 100 "
        "test engines at the canonical configuration, with XGBoost and the true RUL for "
        "reference. DeepSeek-R1, run on 50 engines, is reported in Table 3. A regressor "
        "spreads across the range; the prompted models do not."),
    4: ("The same integer comes out of all of them",
        "Figure 4. Distinct predicted values and modal share on each of the four CMAPSS "
        "sub-datasets, against the number of test engines. Note the logarithmic scale."),
    5: ("Figure 5 breaks these results down by sensor",
        "Figure 5. What the stated direction agrees with, per sensor: the direction the prompt asserts, the direction measured on the training set, the window actually supplied, and the base rate on the same claims. Cued sensors are the ones the prompt names. For s9, s12 and s14 the first two references disagree, and the claims follow the first. Faithfulness and the base rate track each other everywhere. Bars rest on between 10 (s4) and 947 (s11) directional claims."),
    6: ("The data was turned upside down and the explanation stayed the same",
        "Figure 6. The trend-reversal test. For each sensor, the number of (engine, sensor) "
        "pairs whose trend genuinely reversed between the two arms, and the number of those "
        "in which the direction stated by the model reversed with it."),
    7: ("The examples determine where the anchor sits",
        "Figure 7. Mean predicted RUL against the mean RUL of the examples placed in the "
        "prompt, over seven example sets. The dashed line is the test-set mean."),
    8: ("This is the clearest argument for reporting rank correlation",
        "Figure 8. Spearman rank correlation between predicted and true RUL for the 15 FD001 "
        "runs of the main grid, with 95% bootstrap intervals. Random Forest reaches +0.813 "
        "and the repaired LSTM +0.889 on the same engines."),
}


# Alternative text, one entry per figure number. Read by screen readers and
# checked by several publishers' accessibility tools; it should describe what the
# figure shows, not repeat the caption.
ALT_TEXT = {
    1: "Flow diagram of the evaluation protocol: the C-MAPSS sub-datasets feed both the "
       "prompted language models and the supervised baselines, whose outputs are scored on "
       "error, rank correlation, output entropy and explanation faithfulness.",
    2: "Predicted against true remaining useful life for the reference configuration. The "
       "predictions lie on a small number of horizontal lines instead of following the "
       "diagonal.",
    3: "Distribution of predicted remaining useful life on FD001, one series per model, "
       "showing the prompted models concentrated in one or two bins while the reference "
       "distribution spreads across the range.",
    4: "Distinct predicted values and modal share on each of the four C-MAPSS sub-datasets, "
       "against the number of test engines.",
    5: "Per sensor, the share of directional claims that agree with the asserted direction, "
       "the share that agree with the window shown, and the base rate.",
    6: "Counts of paired responses in which the stated sensor direction did or did not follow "
       "the reversal applied to the input trend.",
    7: "Mean predicted remaining useful life against the mean remaining useful life of the "
       "few-shot examples, one point per example set, with a fitted line.",
    8: "Rank correlation with bootstrap intervals for every FD001 configuration of the main "
       "grid, against the supervised baselines on the same engines.",
}

def add_figure(doc_body, after_par, img, caption, width_cm=15.5, alt=None, num=None):
    """Insert a picture paragraph and a caption paragraph after a given one."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    pic_p = OxmlElement("w:p")
    after_par._p.addnext(pic_p)
    from docx.text.paragraph import Paragraph
    pic_par = Paragraph(pic_p, after_par._parent)
    pic_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_par.add_run().add_picture(str(img), width=Cm(width_cm))
    if alt:
        for dp in pic_p.findall(".//" + qn("wp:docPr")):
            dp.set("descr", alt)
            if num:
                dp.set("title", f"Figure {num}")
    # a figure should not be separated from its caption
    pPr = pic_p.get_or_add_pPr()
    if pPr.find(qn("w:keepNext")) is None:
        pPr.append(OxmlElement("w:keepNext"))

    cap_p = OxmlElement("w:p")
    pic_p.addnext(cap_p)
    cap_par = Paragraph(cap_p, after_par._parent)
    cap_par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = cap_par.add_run(caption)
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    return cap_par


def main():
    if not SRC.exists():
        print(f"missing {SRC.name}: run insert_zotero_fields.py first")
        return 1
    doc = Document(SRC)
    placed, missing = [], []

    for num, (anchor, caption) in sorted(FIGURES.items()):
        img = PNG / f"fig{num}.png"
        if not img.exists():
            missing.append(f"fig{num}.png not found")
            continue
        target = None
        for par in doc.paragraphs:
            if anchor.lower() in par.text.lower():
                target = par
                break
        if target is None:
            missing.append(f"figure {num}: anchor not found ({anchor[:40]}...)")
            continue
        add_figure(doc.element.body, target, img, caption,
                   alt=ALT_TEXT.get(num), num=num)
        placed.append(num)

    # Word shows these in File > Info; python-docx leaves its own name there,
    # which should not go to an editor.
    cp = doc.core_properties
    cp.author = "Filippo De Carlo"
    cp.last_modified_by = "Filippo De Carlo"
    cp.title = ("Evaluating large language models for remaining useful life prediction: "
                "a protocol, and what it reveals on CMAPSS")
    cp.comments = ""
    cp.category = ""
    cp.keywords = ("large language models; remaining useful life; prognostics and health "
                   "management; evaluation protocol; explanation faithfulness")

    doc.save(OUT)
    print(f"[saved] {OUT}")
    print(f"figures placed: {placed}")
    for m in missing:
        print(f"  ! {m}")
    ga = PNG / "graphical_abstract.png"
    if ga.exists():
        print(f"\nGraphical abstract is a separate upload: {ga}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
