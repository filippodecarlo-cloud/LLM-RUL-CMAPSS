"""
P3.7 - Fill in the missing bibliographic metadata.

Most cited journal articles in the Zotero library carry no volume, no page range
or article number, and sometimes no DOI, because they were added from a search
result rather than from the record. Elsevier's technical check flags this, and a
reviewer reads it as carelessness.

This does not write to the Zotero database, which would be a bad idea with Zotero
running. It reads the keys the manuscript cites, looks each one up in Crossref by
DOI where there is one and by title otherwise, and writes a list of exactly what
to paste into which field. The author applies it in Zotero and the citations
regenerate.

    python p1/p3_7_reference_metadata.py

Output: results_p1/p3_7_reference_metadata.md
"""
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
P1 = EXP / "results_p1"
ZOTERO = Path(os.path.expanduser(r"~\Zotero\zotero.sqlite"))
MAILTO = "filippo.decarlo@unifi.it"   # Crossref asks for a contact in the user agent

FIELDS = ("title", "date", "publicationTitle", "volume", "issue", "pages",
          "DOI", "url", "proceedingsTitle")


def cited_keys():
    src = (HERE / "insert_zotero_fields.py").read_text(encoding="utf-8")
    return sorted(set(re.findall(r'"([A-Z0-9]{8})"', src)))


def zotero_rows(keys):
    if not ZOTERO.exists():
        raise SystemExit(f"zotero.sqlite not found at {ZOTERO}")
    tmp = Path(tempfile.gettempdir()) / "zotero_readonly.sqlite"
    shutil.copy(ZOTERO, tmp)          # never touch the live database
    con = sqlite3.connect(tmp)
    ph = ",".join("?" * len(keys))
    sel = ",".join(
        f"MAX(CASE WHEN f.fieldName='{f}' THEN v.value END) AS {f.lower()}"
        for f in FIELDS)
    q = f"""SELECT i.key, {sel},
              (SELECT typeName FROM itemTypes WHERE itemTypeID=i.itemTypeID) AS itemtype
            FROM items i
            JOIN itemData d ON d.itemID=i.itemID
            JOIN itemDataValues v ON v.valueID=d.valueID
            JOIN fields f ON f.fieldID=d.fieldID
            WHERE i.key IN ({ph}) GROUP BY i.key"""
    cols = ["key"] + [f.lower() for f in FIELDS] + ["itemtype"]
    return [dict(zip(cols, r)) for r in con.execute(q, keys).fetchall()]


def crossref(doi=None, title=None):
    base = "https://api.crossref.org/works"
    if doi:
        url = f"{base}/{urllib.parse.quote(doi)}"
    elif title:
        url = (f"{base}?rows=1&select=DOI,title,volume,issue,page,article-number,"
               f"container-title,issued,type&query.bibliographic="
               f"{urllib.parse.quote(title)}")
    else:
        return None
    req = urllib.request.Request(
        url, headers={"User-Agent": f"LLM-RUL-CMAPSS/1.0 (mailto:{MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"_error": str(e)}
    msg = data.get("message", {})
    if "items" in msg:
        items = msg["items"]
        if not items:
            return None
        msg = items[0]
    return msg


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    keys = cited_keys()
    rows = {r["key"]: r for r in zotero_rows(keys)}
    print(f"{len(keys)} cited keys, {len(rows)} found in Zotero")

    out = ["# P3.7 - Bibliographic metadata to complete in Zotero\n"]
    out.append("Looked up in Crossref, by DOI where the record has one and by title "
               "otherwise. Apply these in Zotero, not in the bibliography text, then refresh "
               "the citations. A title that Crossref matched only loosely is flagged: check it "
               "before pasting.\n")
    todo = 0
    for key in keys:
        r = rows.get(key)
        if not r:
            out.append(f"## [{key}] not found in the library\n")
            continue
        typ = r["itemtype"]
        need = []
        if typ == "journalArticle":
            if not r["volume"]:
                need.append("volume")
            if not r["pages"]:
                need.append("pages")
            if not r["doi"]:
                need.append("DOI")
        elif typ == "conferencePaper":
            if not r["pages"]:
                need.append("pages")
            if not r["doi"] and not r["url"]:
                need.append("DOI or URL")
        elif typ == "preprint":
            if not r["url"] and not r["doi"]:
                need.append("arXiv identifier")
        if not need:
            continue
        todo += 1
        cr = crossref(doi=r["doi"]) if r["doi"] else crossref(title=r["title"])
        time.sleep(0.4)
        out.append(f"## [{key}] {r['title']}")
        out.append(f"*{typ}, {r['publicationtitle'] or r['proceedingstitle'] or '-'}, "
                   f"{(r['date'] or '')[:4]}* — missing: {', '.join(need)}\n")
        if not cr or "_error" in (cr or {}):
            out.append("Crossref returned nothing"
                       + (f" ({cr['_error']})" if cr and '_error' in cr else "")
                       + "; fill in from the publisher page.\n")
            continue
        ct = (cr.get("container-title") or [""])[0]
        got_title = (cr.get("title") or [""])[0]
        loose = norm(got_title)[:60] != norm(r["title"])[:60]
        page = cr.get("page") or cr.get("article-number")
        vals = [("Volume", cr.get("volume")), ("Issue", cr.get("issue")),
                ("Pages", page), ("DOI", cr.get("DOI")),
                ("Publication", ct)]
        for label, val in vals:
            if val:
                out.append(f"- **{label}**: `{val}`")
        if loose:
            out.append(f"- ⚠ Crossref matched *{got_title}* — verify before pasting.")
        out.append("")
    out.append(f"\n{todo} of {len(keys)} cited references need at least one field.\n")
    P1.mkdir(exist_ok=True)
    (P1 / "p3_7_reference_metadata.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[saved] {P1 / 'p3_7_reference_metadata.md'}  ({todo} entries)")


if __name__ == "__main__":
    sys.exit(main())
