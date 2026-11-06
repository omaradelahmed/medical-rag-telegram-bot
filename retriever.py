"""Step 3: hybrid search (embeddings + keywords) over the index/ folder.
Test from the terminal:  uv run retriever.py "your question"

The index is reloaded automatically when ingest.py adds / changes / removes a document,
so the running bot / MCP server never needs a restart.
"""
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

import store

EMBED_MODEL = "bge-m3"
KEEP_ALIVE = "30m"  # keep the embedding model in memory between questions
RRF_K = 60          # reciprocal rank fusion constant

# The books say "clinical picture", users say "clinical features": swap to the book's wording.
# Ordered from most specific to least; only the first match is used.
SYNONYMS = [
    ("clinical signs and symptoms", "clinical picture"),
    ("signs and symptoms", "clinical picture"),
    ("symptoms and signs", "clinical picture"),
    ("clinical manifestations", "clinical picture"),
    ("clinical features", "clinical picture"),
    ("clinical signs", "clinical picture"),
    ("manifestations", "clinical picture"),
    ("symptoms", "clinical picture"),
    ("diagnosis", "investigations"),
    ("management", "treatment"),
]
# The books use abbreviations heavily. Extend this list as you find misses.
ABBREVIATIONS = {
    "chronic kidney disease": "CKD",
    "acute kidney injury": "AKI",
    "chronic renal failure": "CRF",
    "myocardial infarction": "MI",
    "heart failure": "HF",
    "atrial fibrillation": "AF",
    "diabetes mellitus": "DM",
    "diabetic ketoacidosis": "DKA",
    "chronic obstructive pulmonary disease": "COPD",
    "pulmonary embolism": "PE",
    "infective endocarditis": "IE",
    "systemic lupus erythematosus": "SLE",
    "rheumatoid arthritis": "RA",
    "inflammatory bowel disease": "IBD",
    "tuberculosis": "TB",
    "idiopathic thrombocytopenic purpura": "ITP",
}
STOPWORDS = {"of", "the", "and", "in", "a", "an", "to", "for", "is", "are",
             "with", "by", "on", "or", "what", "how", "does", "do"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class _Index:
    stamp: tuple = ()
    records: list = field(default_factory=list)  # [{"source","owner","page","text"}]
    matrix: np.ndarray | None = None              # normalized embeddings
    bm25: BM25Okapi | None = None


_index: _Index | None = None
_lock = threading.Lock()
_page_cache: dict[tuple[str, int], str | None] = {}


def _build() -> _Index:
    stamp = store.stamp()  # taken BEFORE loading: a change during loading triggers another reload
    records, mats = [], []
    for meta, emb in store.load_all():
        for c in meta["chunks"]:
            records.append({"source": meta["source"], "owner": meta.get("owner", "shared"),
                            "page": c["page"], "text": c["text"]})
        mats.append(emb)
    if not records:
        return _Index(stamp=stamp)
    matrix = np.vstack(mats).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    bm25 = BM25Okapi([_tokenize(r["text"]) for r in records])
    return _Index(stamp, records, matrix, bm25)


def _current() -> _Index:
    """The up-to-date index (reloads only if the index folder changed)."""
    global _index
    if _index is None or _index.stamp != store.stamp():
        with _lock:
            if _index is None or _index.stamp != store.stamp():
                _index = _build()
    return _index


def __getattr__(name):  # keeps `retriever._records` working (used by the server and debug_search)
    if name == "_records":
        return _current().records
    raise AttributeError(name)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """rank 0 = best score."""
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks


def expand_query(question: str) -> list[str]:
    """Return the question plus a few rewrites that match the books' wording."""
    low = question.strip().strip("\"'").lower()
    variants = [low]
    for old, new in SYNONYMS:
        if old in low:
            variants.append(low.replace(old, new))
            break
    for v in list(variants):
        for full, abbr in ABBREVIATIONS.items():
            if full in v:
                variants.append(v.replace(full, abbr))
    return list(dict.fromkeys(variants))[:6]


def _docs_dir() -> Path:
    return Path(__file__).parent / "docs"


def get_page(source: str, page: int) -> str | None:
    """Return the FULL text of one PDF page (1-based), not a 1200-char search chunk.

    Used when a big table / list on a page got truncated by TEXT_LIMIT in a search
    chunk. Re-reads the PDF on demand; cached per index stamp, so re-indexing
    refreshes it automatically. Returns None for out-of-range pages.
    """
    idx = _current()
    stamp = idx.stamp
    key = (source, page, stamp)
    if key in _page_cache:
        return _page_cache[key]

    # guard against arbitrary source strings: only indexed sources are readable
    if not any(r["source"] == source for r in idx.records):
        raise FileNotFoundError(f"source '{source}' is not indexed")
    if not 1 <= page <= 100000:
        raise ValueError("page must be a positive integer")
    if len(_page_cache) > 500:  # bound the cache
        _page_cache.clear()

    from pypdf import PdfReader  # lazy: don't import a PDF lib at module load
    path = _docs_dir() / source
    if not path.exists():
        raise FileNotFoundError(f"file not found: {source}")
    reader = PdfReader(str(path))
    if page > len(reader.pages):
        return None
    text = reader.pages[page - 1].extract_text() or ""
    _page_cache[key] = text
    return text


def search(question: str, top_k: int = 5) -> list[dict]:
    """Hybrid search: embeddings + BM25 keywords, fused with reciprocal rank fusion.
    The returned `score` is the embedding cosine similarity (used as the relevance threshold)."""
    idx = _current()
    if not idx.records:
        return []
    queries = expand_query(question)
    embs = np.array(
        ollama.embed(model=EMBED_MODEL, input=queries, keep_alive=KEEP_ALIVE).embeddings,
        dtype=np.float32,
    )
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    dense = (idx.matrix @ embs.T).max(axis=1)  # cosine similarity, best variant per chunk
    keyword = np.max([idx.bm25.get_scores(_tokenize(q)) for q in queries], axis=0)

    fused = 1.0 / (RRF_K + _ranks(dense)) + 1.0 / (RRF_K + _ranks(keyword))
    best = np.argsort(-fused)[:top_k]
    return [
        {
            "source": idx.records[i]["source"],
            "page": idx.records[i]["page"],
            "text": idx.records[i]["text"],
            "score": float(dense[i]),
        }
        for i in best
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    print("Variants:", expand_query(question), "\n")
    for r in search(question):
        print(f"[{r['score']:.3f}] {r['source']} p.{r['page']}")
        print(f"   {r['text'][:200]}...\n")
