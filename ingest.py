"""Step 2: build / update the search index from the PDFs in docs/.

  uv run ingest.py                  add new PDFs, re-index changed ones, drop deleted ones
  uv run ingest.py --force          re-embed everything
  uv run ingest.py --list           show what is indexed
  uv run ingest.py --remove NAME    remove one source from the index

Reusable from other code (e.g. a future "send a PDF to the bot" feature):
    import ingest
    result = await asyncio.to_thread(ingest.ingest_pdf, path)   # blocking -> run in a thread
The search side (retriever.py) notices index changes by itself: no restart needed.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

import store

DOCS_DIR = Path(__file__).parent / "docs"
LEGACY_FILE = Path(__file__).parent / "index.json"  # old single-file index
EMBED_MODEL = "bge-m3"
CHUNK_SIZE = 700       # characters (smaller chunks = more precise search)
OVERLAP = 100
HEADER_MAX = 400       # max length of the page-titles block
HEADER_FALLBACK = 150  # used when no titles block is found
MIN_PAGE_CHARS = 50    # skip pages with (almost) no text, e.g. image-only pages
BATCH = 16

# If any of these settings change, documents are re-indexed automatically.
CONFIG = {"format": 2, "embed_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE, "overlap": OVERLAP,
          "header_max": HEADER_MAX, "header_fallback": HEADER_FALLBACK, "min_page_chars": MIN_PAGE_CHARS}

# Study Smarter pages list ALL topic titles first, then the first exam tag
# (e.g. "Nov 2018", "2023-24"). Everything before that tag = the page's titles.
TAG_RE = re.compile(r"(?:Nov|May|Jun|Jul) 20\d\d|2023–24")


def page_header(text: str) -> str:
    """Titles of all topics on the page (repeated in every chunk of the page)."""
    m = TAG_RE.search(text)
    if m and 20 <= m.start() <= HEADER_MAX:
        return text[:m.start()].strip()
    return text[:HEADER_FALLBACK]


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= CHUNK_SIZE:
        return [text]
    header = page_header(text)
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        # the first chunk already starts with the header; add it to the others
        chunks.append(piece if start == 0 else f"{header} ... {piece}")
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_name(path: Path) -> str:
    """Name used in the index and in 'Sources:' (relative to docs/ when possible)."""
    try:
        return Path(path).resolve().relative_to(DOCS_DIR.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def extract_chunks(path: Path) -> tuple[list[dict], int, int]:
    """-> (chunks [{"page","text"}], pages_with_text, pages_skipped)"""
    reader = PdfReader(str(path))
    chunks, kept, skipped = [], 0, 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) < MIN_PAGE_CHARS:
            skipped += 1
            continue
        kept += 1
        chunks += [{"page": page_no, "text": c} for c in chunk_text(text)]
    return chunks, kept, skipped


def embed_texts(texts: list[str], progress=print) -> np.ndarray:
    out = []
    for i in range(0, len(texts), BATCH):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i:i + BATCH])
        out.extend(resp.embeddings)
        if (i // BATCH) % 25 == 0:
            progress(f"    embedded {min(i + BATCH, len(texts))}/{len(texts)}")
    return np.array(out, dtype=np.float32)


def ingest_pdf(path, force: bool = False, owner: str = "shared", progress=print) -> dict:
    """Index one PDF. Returns {"status": added|updated|unchanged|no_text, "source", "chunks"}.
    Blocking (embeddings take a while): call it with asyncio.to_thread from async code."""
    path = Path(path)
    source = source_name(path)
    file_hash = file_sha256(path)
    with store.IngestLock():
        existing = store.read_info(source)
        if existing and not force and existing["file_hash"] == file_hash and existing["config"] == CONFIG:
            return {"status": "unchanged", "source": source, "chunks": existing["n_chunks"]}

        chunks, kept, skipped = extract_chunks(path)
        if not chunks:  # e.g. a scanned PDF: pypdf finds no text
            return {"status": "no_text", "source": source, "chunks": 0}
        progress(f"    {kept} pages with text, {skipped} skipped, {len(chunks)} chunks")
        embeddings = embed_texts([c["text"] for c in chunks], progress)
        store.save_source(source, file_hash, CONFIG, chunks, embeddings, owner=owner)
    return {"status": "updated" if existing else "added", "source": source, "chunks": len(chunks)}


def remove_source(source: str) -> bool:
    with store.IngestLock():
        return store.delete_source(source)


def migrate_legacy_index() -> int:
    """Old index.json -> index/ folder WITHOUT re-embedding (assumes the same settings as now)."""
    if not LEGACY_FILE.exists() or list(store.INDEX_DIR.glob("*.npz")):
        return 0
    print(f"Migrating {LEGACY_FILE.name} to the new per-document index (no re-embedding) ...")
    by_source = defaultdict(list)
    for r in json.loads(LEGACY_FILE.read_text(encoding="utf-8")):
        by_source[r["source"]].append(r)
    done = 0
    with store.IngestLock():
        for source, recs in by_source.items():
            pdf = DOCS_DIR / source
            if not pdf.exists():
                print(f"  ! {source}: not found in docs/, skipped (it will be re-indexed if you add it back)")
                continue
            chunks = [{"page": r["page"], "text": r["text"]} for r in recs]
            emb = np.array([r["embedding"] for r in recs], dtype=np.float32)
            store.save_source(source, file_sha256(pdf), CONFIG, chunks, emb)
            print(f"  {source}: {len(chunks)} chunks")
            done += 1
    LEGACY_FILE.rename(LEGACY_FILE.with_suffix(".json.bak"))
    print(f"  old file kept as {LEGACY_FILE.name}.bak (you can delete it)")
    return done


def sync_docs(force: bool = False) -> None:
    migrate_legacy_index()
    DOCS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
    seen = set()
    for pdf in pdfs:
        name = source_name(pdf)
        seen.add(name)
        print(f"{name}")
        r = ingest_pdf(pdf, force=force)
        label = {"added": "ADDED", "updated": "UPDATED", "unchanged": "unchanged (skipped)",
                 "no_text": "NO TEXT (scanned PDF? needs OCR)"}[r["status"]]
        print(f"  -> {label}, {r['chunks']} chunks")
    for gone in sorted(set(store.list_sources()) - seen):
        remove_source(gone)
        print(f"{gone}\n  -> REMOVED (file no longer in docs/)")
    total = sum(i["n_chunks"] for i in store.list_sources().values())
    print(f"\nIndex: {len(store.list_sources())} documents, {total} chunks")


def main():
    ap = argparse.ArgumentParser(description="Build or update the search index.")
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--list", action="store_true", help="show indexed documents")
    ap.add_argument("--remove", metavar="NAME", help="remove one source (as shown by --list)")
    args = ap.parse_args()

    if args.list:
        migrate_legacy_index()
        for src, info in store.list_sources().items():
            print(f"{src}: {info['n_chunks']} chunks, owner={info.get('owner')}")
    elif args.remove:
        print("removed" if remove_source(args.remove) else "not found in the index")
    else:
        sync_docs(force=args.force)


if __name__ == "__main__":
    main()
