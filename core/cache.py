"""
Parses + embeds each CV ONCE and keeps the result on disk (cache/cv_index.pkl).

A new job description never triggers re-parsing: every search only embeds the
JD and compares it with what is already stored. Only CVs that are new or
changed (per source.get_cv_version) are processed. Progress is checkpointed,
so an interrupted first run resumes where it stopped.

CVs whose text can't be extracted are stored as unreadable: they are NOT
ranked (an empty text would rank randomly) but they are listed so nobody is
lost silently.
"""

import os
import pickle
import logging
import numpy as np

from core.embedder import chunk_text, embed_chunks, embedding_dim
from core.parser import is_extraction_suspect

logger = logging.getLogger("cv_parser")

CACHE_PATH = os.path.join("cache", "cv_index.pkl")
CACHE_VERSION = 3  # bump when the stored format or chunking changes


class CVIndex:
    """Everything the matcher needs, as flat numpy arrays."""

    def __init__(self, cv_ids, E, starts, doc_vecs, readable, texts):
        self.cv_ids = cv_ids          # list of CV ids
        self.E = E                    # (total_chunks, dim) every chunk of every CV
        self.starts = starts          # (n_cvs,) first row of each CV inside E
        self.doc_vecs = doc_vecs      # (n_cvs, dim) one normalized vector per CV
        self.readable = readable      # (n_cvs,) False = text could not be extracted
        self.texts = texts            # {cv_id: extracted text}


def _load() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == CACHE_VERSION:
                return data["cvs"]
            logger.info("Cache format changed — rebuilding index from scratch")
        except Exception as e:
            logger.error(f"Cache unreadable ({e}) — rebuilding index from scratch")
    return {}


def _save(cvs: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "cvs": cvs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE_PATH)  # atomic: a crash mid-save can't corrupt the cache


def get_index(source, on_progress=None, save_every: int = 250,
              retry_unreadable: bool = False) -> CVIndex:
    """
    retry_unreadable: also re-attempt CVs that failed extraction last time
    (build_index.py turns this on, so fixing the parser and re-running picks them
    up; the app leaves it off so a search never waits on slow OCR / Word retries).
    """
    cvs = _load()

    cv_ids, todo = [], []
    for cid in source.list_cv_ids():
        try:
            version = source.get_cv_version(cid)
        except OSError as e:
            logger.error(f"'{cid}': cannot access file ({e}) — left out of the index")
            continue
        cv_ids.append(cid)
        entry = cvs.get(cid)
        if (entry is None or entry["version"] != version
                or (retry_unreadable and not entry["readable"])):
            todo.append((cid, version))

    dim = embedding_dim()
    for n, (cid, version) in enumerate(todo, 1):
        try:
            text = source.get_cv_text(cid)
        except Exception as e:  # corrupt / password-protected / unsupported
            logger.error(f"'{cid}': extraction failed ({type(e).__name__}: {e}) — REVIEW MANUALLY")
            text = ""

        readable = not is_extraction_suspect(text)
        emb = embed_chunks(chunk_text(text)) if readable else np.zeros((1, dim), dtype=np.float32)
        cvs[cid] = {"version": version, "text": text, "emb": emb, "readable": readable}

        if n % save_every == 0:
            _save(cvs)
        if on_progress:
            on_progress(n, len(todo))

    stale = set(cvs) - set(cv_ids)  # files that were deleted or renamed
    for cid in stale:
        del cvs[cid]
    if todo or stale:
        _save(cvs)

    if not cv_ids:
        return CVIndex([], np.zeros((0, dim), np.float32), np.zeros(0, dtype=np.int64),
                       np.zeros((0, dim), np.float32), np.zeros(0, dtype=bool), {})

    parts, starts, offset = [], [], 0
    for cid in cv_ids:
        emb = cvs[cid]["emb"]
        starts.append(offset)
        offset += len(emb)
        parts.append(emb)
    E = np.vstack(parts)
    starts = np.array(starts, dtype=np.int64)

    sums = np.add.reduceat(E, starts, axis=0)
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_vecs = (sums / norms).astype(np.float32)

    readable = np.array([cvs[cid]["readable"] for cid in cv_ids], dtype=bool)
    texts = {cid: cvs[cid]["text"] for cid in cv_ids}
    return CVIndex(cv_ids, E, starts, doc_vecs, readable, texts)