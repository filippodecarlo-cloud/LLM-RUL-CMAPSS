"""
P3.8 - Write the missing bibliographic fields into Zotero.

Companion to p3_7_reference_metadata.py, which looks the values up in Crossref
and writes them to a markdown file. This applies that file to the local Zotero
database, which is otherwise about seventy fields of manual copying.

Two rules make it safe to run:

  * it only ever fills a field that is EMPTY. An existing value is never
    overwritten, so a correction the author made by hand cannot be undone.
  * it refuses to run while Zotero is open, and it takes a timestamped copy of
    the database before touching it.

Zotero must be closed. Use --dry-run first; it prints exactly what would change
and writes nothing.

    python p1/p3_8_write_zotero_fields.py --dry-run
    python p1/p3_8_write_zotero_fields.py --apply
"""
import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SRC = EXP / "results_p1" / "p3_7_reference_metadata.md"
ZOTERO = Path(os.path.expanduser(r"~\Zotero\zotero.sqlite"))

# label in the markdown -> Zotero field name
FIELD_MAP = {"Volume": "volume", "Pages": "pages", "DOI": "DOI", "Issue": "issue"}
# deliberately not mapped: "Publication". Changing a journal or proceedings name
# is an editorial decision, not a gap to be filled mechanically.


def zotero_running():
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=20).stdout
        return "zotero" in out.lower()
    except Exception:
        return False


def parse(md):
    """[{key, title, fields: {zotero_field: value}}] from the markdown report."""
    out = []
    cur = None
    for line in md.splitlines():
        m = re.match(r"^## \[([A-Z0-9]{8})\]\s*(.*)$", line)
        if m:
            cur = {"key": m.group(1), "title": m.group(2).strip(), "fields": {},
                   "loose": False}
            out.append(cur)
            continue
        if cur is None:
            continue
        if line.lstrip().startswith("- ⚠"):
            # p3_7 flags a record whose Crossref match was only approximate. The
            # values under such a heading may belong to a different paper, so the
            # record is dropped rather than written. This is not hypothetical: the
            # first run of this script wrote an EMNLP DOI and page range into an
            # arXiv preprint because it read the values and ignored the warning.
            cur["loose"] = True
            continue
        m = re.match(r"^- \*\*([A-Za-z]+)\*\*:\s*`(.+?)`\s*$", line)
        if m and m.group(1) in FIELD_MAP:
            cur["fields"][FIELD_MAP[m.group(1)]] = m.group(2).strip()
    for r in out:
        if r.get("loose"):
            print(f"[{r['key']}] SKIPPED: Crossref match flagged as approximate. "
                  f"Check this one by hand.")
    return [r for r in out if r["fields"] and not r.get("loose")]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"missing {SRC}; run p3_7_reference_metadata.py first")
    if not ZOTERO.exists():
        raise SystemExit(f"zotero.sqlite not found at {ZOTERO}")
    if zotero_running():
        raise SystemExit("Zotero is running. Close it and try again.")

    records = parse(SRC.read_text(encoding="utf-8"))
    print(f"{len(records)} records in the report\n")

    if args.apply:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = ZOTERO.with_name(f"zotero.sqlite.backup-{stamp}")
        shutil.copy(ZOTERO, backup)
        print(f"backup: {backup.name}\n")

    con = sqlite3.connect(ZOTERO)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    fid = {n: cur.execute("SELECT fieldID FROM fields WHERE fieldName=?", (n,)).fetchone()[0]
           for n in set(FIELD_MAP.values())}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    written = skipped = missing = 0
    for rec in records:
        row = cur.execute(
            "SELECT itemID, itemTypeID FROM items WHERE key=?", (rec["key"],)).fetchone()
        if not row:
            print(f"[{rec['key']}] NOT IN LIBRARY - skipped")
            missing += 1
            continue
        item_id, type_id = row
        valid = {r[0] for r in cur.execute(
            """SELECT f.fieldName FROM itemTypeFields itf
               JOIN fields f ON f.fieldID=itf.fieldID WHERE itf.itemTypeID=?""", (type_id,))}
        changes = []
        for field, value in rec["fields"].items():
            if field not in valid:
                changes.append(("not valid for this item type", field, value))
                continue
            existing = cur.execute(
                """SELECT v.value FROM itemData d JOIN itemDataValues v ON v.valueID=d.valueID
                   WHERE d.itemID=? AND d.fieldID=?""", (item_id, fid[field])).fetchone()
            if existing and str(existing[0]).strip():
                changes.append((f"already set to {existing[0]!r}", field, value))
                skipped += 1
                continue
            changes.append(("WRITE", field, value))
            if args.apply:
                vid = cur.execute(
                    "SELECT valueID FROM itemDataValues WHERE value=?", (value,)).fetchone()
                if vid:
                    vid = vid[0]
                else:
                    cur.execute("INSERT INTO itemDataValues (value) VALUES (?)", (value,))
                    vid = cur.lastrowid
                cur.execute(
                    "INSERT OR REPLACE INTO itemData (itemID, fieldID, valueID) VALUES (?,?,?)",
                    (item_id, fid[field], vid))
            written += 1
        if args.apply and any(c[0] == "WRITE" for c in changes):
            # mark the item as locally modified so Zotero re-syncs it
            cur.execute("UPDATE items SET dateModified=?, clientDateModified=?, synced=0 "
                        "WHERE itemID=?", (now, now, item_id))
        print(f"[{rec['key']}] {rec['title'][:66]}")
        for what, field, value in changes:
            mark = "  +" if what == "WRITE" else "  ."
            print(f"{mark} {field:8s} = {value:38s} {'' if what == 'WRITE' else what}")
        print()

    if args.apply:
        con.commit()
    con.close()

    verb = "written" if args.apply else "would be written"
    print(f"{written} fields {verb}, {skipped} left alone (already set), "
          f"{missing} keys not found")
    if not args.apply:
        print("\nNothing was changed. Re-run with --apply to write.")


if __name__ == "__main__":
    sys.exit(main())
