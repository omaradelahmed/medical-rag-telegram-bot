"""Build quiz_bank.json from the "past questions" PDF: question TITLES only (no answers).

  uv add pymupdf                      (once)
  uv run build_quiz.py                (uses docs/Study_Smarter_IM_Paper1.pdf)
  uv run build_quiz.py path/to.pdf    (another PDF with the same style)

How it works: in this book every question title is BOLD, 14.5 pt, in a coloured (not white)
font, e.g. "Causes of secondary hypertension". Section names (CARDIOLOGY...) are bold uppercase
lines. Reading the font style is exact and free: no LLM and no guessing.
Each entry also keeps `page`, `y` and `y_end` (where the title sits on the page), so a future
/answer feature can cut the answer text out of the PDF without re-parsing everything.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

HERE = Path(__file__).parent
DEFAULT_PDF = HERE / "docs" / "Study_Smarter_IM_Paper1.pdf"
OUT_FILE = HERE / "quiz_bank.json"

TITLE_SIZE = 14.5      # point size of question titles in this book
SIZE_TOLERANCE = 0.3
WHITE = 0xFFFFFF       # white bold text = small sub-labels on coloured banners (Definition, Diagnosis...)
TAG_RE = re.compile(r"(?:Nov|May|Jun|Jul)\s+20\d\d|2023–24")  # exam-session tags: "Nov 2018", "2023-24"


def read_lines(page) -> list[dict]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            s = line["spans"][0]
            lines.append({
                "y0": line["bbox"][1], "y1": line["bbox"][3], "x": line["bbox"][0],
                "size": round(s["size"], 1),
                "bold": bool(s["flags"] & 16) or "Bold" in s["font"],
                "color": s["color"], "text": text,
            })
    return sorted(lines, key=lambda d: (d["y0"], d["x"]))


def is_title(l: dict) -> bool:
    return l["bold"] and abs(l["size"] - TITLE_SIZE) <= SIZE_TOLERANCE and l["color"] != WHITE


def build(pdf_path: Path) -> list[dict]:
    source = pdf_path.name
    doc = pymupdf.open(str(pdf_path))
    items, section = [], ""
    for page_no, page in enumerate(doc, start=1):
        lines = read_lines(page)
        # section name: bold, uppercase, bigger than body text, and not a question title
        for l in lines:
            if l["bold"] and l["size"] >= 14.0 and not is_title(l) and l["text"].isupper() and len(l["text"]) <= 20:
                section = l["text"]
                break

        # question titles (a long title can wrap onto the next line: merge those)
        titles = []
        for l in lines:
            if not is_title(l):
                continue
            if titles and l["y0"] - titles[-1]["y1"] < 6:  # directly under the previous title line
                titles[-1]["text"] += " " + l["text"]
                titles[-1]["y1"] = l["y1"]
            else:
                titles.append({"text": l["text"], "y0": l["y0"], "y1": l["y1"], "tags": []})

        # exam-session tags ("Nov 2018 · 2023-24") belong to the closest title above them
        for l in lines:
            if l["bold"] or not (10.0 <= l["size"] <= 12.0) or not TAG_RE.search(l["text"]):
                continue
            above = [t for t in titles if t["y0"] <= l["y0"]]
            if above:
                for tag in (p.strip() for p in l["text"].split("·")):
                    if tag and tag not in above[-1]["tags"]:
                        above[-1]["tags"].append(tag)

        for i, t in enumerate(titles):
            items.append({
                "id": hashlib.sha1(f"{source}|{page_no}|{t['text']}".encode()).hexdigest()[:10],
                "source": source,
                "page": page_no,
                "section": section,
                "question": t["text"],
                "asked_in": t["tags"],
                "y": round(t["y0"]),
                "y_end": round(titles[i + 1]["y0"]) if i + 1 < len(titles) else round(page.rect.height),
            })
    return items


def main():
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    items = build(pdf)
    if not items:
        raise SystemExit("No question titles found: is this the same book style?")
    OUT_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    per_section = {}
    for q in items:
        per_section[q["section"]] = per_section.get(q["section"], 0) + 1
    print(f"{pdf.name}: {len(items)} questions -> {OUT_FILE.name}")
    for sec, n in per_section.items():
        print(f"  {sec or '(no section)'}: {n}")


if __name__ == "__main__":
    main()
