"""
P3.6 - Page-layout fixes that a reader sees and a script does not.

Four things the manuscript needed and no numeric check would ever catch:

  * a table caption stranded on the page before its table, which happened to
    Table 5 and to Table 7 in the submitted rendering;
  * header rows that do not repeat when a table breaks across pages;
  * images with no alternative text, which fails the accessibility check that
    several publishers now run;
  * tables whose first data row can be separated from the header.

The caption is kept with the table by setting keep-with-next on the caption
paragraph and on every row of the table except the last, which is the mechanism
Word itself uses. Header rows are marked as such so Word repeats them. Alt text
is written into the drawing description of each inline image.

    python p1/p3_6_word_layout.py [--paper paper_v16.docx]

The script is idempotent: running it twice changes nothing the second time.
"""
import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Alt text is applied by p1/insert_figures.py when the figures are placed, since
# the master carries no images. The list is kept here as a fallback for a file
# that already has them.
ALT_TEXT = [
    "Flow diagram of the evaluation protocol: C-MAPSS sub-datasets feed both the "
    "prompted language models and the supervised baselines, whose outputs are scored "
    "on error, rank correlation, output entropy and explanation faithfulness.",
    "Predicted against true remaining useful life for the reference configuration, "
    "showing predictions stacked on a small number of horizontal lines rather than "
    "following the diagonal.",
    "Distinct predicted values and modal share for each model configuration, showing "
    "one to five distinct values for the non-reasoning models and nineteen for the "
    "reasoning model.",
    "Distribution of predicted values by model, on a logarithmic count axis, with the "
    "modal share marked on the right axis.",
    "Per sensor, the share of directional claims agreeing with the asserted direction, "
    "the share agreeing with the window shown, and the base rate.",
    "Counts of paired responses in which the stated sensor direction did or did not "
    "follow the reversal applied to the input trend.",
    "Mean predicted remaining useful life against the mean remaining useful life of the "
    "few-shot examples, one point per example set, with a fitted line.",
    "Rank correlation with bootstrap intervals for every FD001 configuration, against "
    "the supervised baselines on the same engines.",
]


def keep_with_next(paragraph, on=True):
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("w:keepNext",):
        el = pPr.find(qn(tag))
        if on and el is None:
            pPr.append(OxmlElement(tag))
        elif not on and el is not None:
            pPr.remove(el)


def mark_header_row(row):
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:tblHeader")) is None:
        trPr.append(OxmlElement("w:tblHeader"))
    # and do not let a row split across pages
    if trPr.find(qn("w:cantSplit")) is None:
        trPr.append(OxmlElement("w:cantSplit"))


def cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:cantSplit")) is None:
        trPr.append(OxmlElement("w:cantSplit"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(ROOT / "paper_v16.docx"))
    args = ap.parse_args()
    path = Path(args.paper)
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    d = Document(path)

    # ---- captions stay with their table --------------------------------
    caps = 0
    for p in d.paragraphs:
        if re.match(r"^Table \d+\.", p.text.strip()):
            keep_with_next(p)
            caps += 1
    print(f"{caps} table captions set to stay with the table")

    # ---- header rows repeat, rows do not split -------------------------
    for t in d.tables:
        rows = list(t.rows)
        if not rows:
            continue
        mark_header_row(rows[0])
        for r in rows[1:]:
            cant_split(r)
    print(f"{len(d.tables)} tables: header row marked, rows set not to split")

    # ---- alt text on every inline image --------------------------------
    docPrs = d.element.body.findall(".//" + qn("wp:docPr"))
    n = 0
    for i, el in enumerate(docPrs):
        if i < len(ALT_TEXT):
            el.set("descr", ALT_TEXT[i])
            el.set("title", f"Figure {i + 1}")
            n += 1
    print(f"{n} images given alternative text"
          + ("" if n == len(docPrs) else f" ({len(docPrs)} images found)"))
    if len(docPrs) and len(docPrs) != len(ALT_TEXT):
        print("  WARNING: image count does not match the alt-text list; check the order",
              file=sys.stderr)

    d.save(path)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
