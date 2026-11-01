"""Index storage: ONE .npz file per source document (embeddings + metadata).

index/<id>.npz  ->  embeddings (float32 matrix) + meta (JSON: source, owner, file_hash, config, chunks)

Why one file per document:
  * adding / updating / removing a document touches only its own file
  * writes are atomic (temp file + os.replace), so a reader never sees a half-written file
  * readers can notice changes cheaply through stamp() and reload without a restart
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator

import numpy as np

INDEX_DIR = Path(__file__).parent / "index"


def _path_for(source: str) -> Path:
    return INDEX_DIR / (hashlib.sha1(source.encode("utf-8")).hexdigest()[:16] + ".npz")


def save_source(source: str, file_hash: str, config: dict, chunks: list[dict],
                embeddings, owner: str = "shared") -> None:
    """chunks: [{"page": int, "text": str}, ...]; embeddings: (len(chunks), dim). Atomic write."""
    INDEX_DIR.mkdir(exist_ok=True)
    meta = {"source": source, "owner": owner, "file_hash": file_hash,
            "config": config, "chunks": chunks}
    fd, tmp = tempfile.mkstemp(dir=INDEX_DIR, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, embeddings=np.asarray(embeddings, dtype=np.float32),
                     meta=np.array(json.dumps(meta, ensure_ascii=False)))
        target = _path_for(source)
        for attempt in range(10):  # Windows: the file may be open for a moment in a reader
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.3)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read(path: Path, with_embeddings: bool):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return (meta, z["embeddings"]) if with_embeddings else meta


def read_info(source: str) -> dict | None:
    """Metadata of an indexed source (file_hash, config, n_chunks...) or None."""
    path = _path_for(source)
    if not path.exists():
        return None
    meta = _read(path, False)
    return {**{k: v for k, v in meta.items() if k != "chunks"}, "n_chunks": len(meta["chunks"])}


def list_sources() -> dict[str, dict]:
    out = {}
    for path in sorted(INDEX_DIR.glob("*.npz")):
        meta = _read(path, False)
        out[meta["source"]] = {**{k: v for k, v in meta.items() if k != "chunks"},
                               "n_chunks": len(meta["chunks"])}
    return out


def load_all() -> Iterator[tuple[dict, np.ndarray]]:
    for path in sorted(INDEX_DIR.glob("*.npz")):
        yield _read(path, True)


def delete_source(source: str) -> bool:
    path = _path_for(source)
    if path.exists():
        path.unlink()
        return True
    return False


def stamp() -> tuple:
    """Cheap fingerprint of the index folder; changes whenever a document is added/changed/removed."""
    if not INDEX_DIR.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                        for p in INDEX_DIR.glob("*.npz")))


class IngestLock:
    """Only one ingest at a time (CLI or bot). Exclusive-create lock file."""
    path = INDEX_DIR / ".lock"

    def __enter__(self):
        INDEX_DIR.mkdir(exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("Another ingest is running. If none is, delete index/.lock") from None
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
