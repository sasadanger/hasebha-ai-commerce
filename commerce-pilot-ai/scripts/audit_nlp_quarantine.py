"""Lightweight, model-free audits for quarantined NLP tabular files."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ARABIC = re.compile(r"[\u0600-\u06ff]")
LATIN = re.compile(r"[A-Za-z]")
URL = re.compile(r"https?://|www\.", re.I)
MENTION = re.compile(r"(?<!\w)@[\w_]+")
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def xlsx_rows(path: Path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iterfind(".//m:t", NS)) for si in root]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        sheet = wb.find(".//m:sheet", NS)
        target = relmap[sheet.attrib["{%s}id" % NS["r"]]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(z.read(target))
        for row in root.iterfind(".//m:row", NS):
            values = []
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                value = "" if v is None else (v.text or "")
                if c.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                elif c.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in c.iterfind(".//m:t", NS))
                values.append(value)
            yield values


def delimited_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        yield from csv.reader(f, delimiter="\t")


def audit(path: Path, text_column: str | None, label_columns: list[str]):
    rows = list(xlsx_rows(path) if path.suffix.lower() == ".xlsx" else delimited_rows(path))
    header = [str(x).strip() for x in rows[0]]
    body = [r + [""] * (len(header) - len(r)) for r in rows[1:]]
    lower = [h.lower() for h in header]
    if text_column:
        ti = lower.index(text_column.lower())
    else:
        candidates = [i for i, h in enumerate(lower) if any(k in h for k in ("text", "tweet", "review", "comment"))]
        ti = candidates[0] if candidates else max(range(len(header)), key=lambda i: statistics.mean([len(r[i]) for r in body[:100] or [[""]]]))
    lis = [lower.index(x.lower()) for x in label_columns if x.lower() in lower]
    texts = [str(r[ti]).strip() for r in body]
    labels = [tuple(str(r[i]).strip() for i in lis) for r in body]
    by_text = defaultdict(set)
    for t, lab in zip(texts, labels):
        if t:
            by_text[t].add(lab)
    lengths = [len(t) for t in texts]
    n = len(texts) or 1
    return {
        "path": path.as_posix(), "row_count": len(body), "columns": header,
        "text_column": header[ti], "label_columns": [header[i] for i in lis],
        "missing_or_empty_text": sum(not t for t in texts),
        "exact_duplicate_text_rows": len(texts) - len(set(texts)),
        "conflicting_label_duplicate_texts": sum(len(v) > 1 for v in by_text.values()),
        "text_length": {"min": min(lengths, default=0), "median": statistics.median(lengths) if lengths else 0,
                        "mean": statistics.mean(lengths) if lengths else 0, "max": max(lengths, default=0)},
        "arabic_script_rate": sum(bool(ARABIC.search(t)) for t in texts) / n,
        "latin_script_rate": sum(bool(LATIN.search(t)) for t in texts) / n,
        "mixed_script_rate": sum(bool(ARABIC.search(t)) and bool(LATIN.search(t)) for t in texts) / n,
        "url_rate": sum(bool(URL.search(t)) for t in texts) / n,
        "mention_rate": sum(bool(MENTION.search(t)) for t in texts) / n,
        "email_like_rate": sum(bool(EMAIL.search(t)) for t in texts) / n,
        "phone_like_rate": sum(bool(PHONE.search(t)) for t in texts) / n,
        "label_counts": {"|".join(k): v for k, v in Counter(labels).items()},
        "encoding_replacement_characters": sum(t.count("\ufffd") for t in texts),
        "pii_risk": "MEDIUM_FREE_TEXT_REVIEW_REQUIRED",
        "models_or_embeddings_used": False,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--text-column")
    ap.add_argument("--label-column", action="append", default=[])
    args = ap.parse_args()
    print(json.dumps(audit(args.path, args.text_column, args.label_column), ensure_ascii=False, indent=2))
