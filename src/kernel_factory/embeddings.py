"""Pluggable text embedding.

Default backend is a deterministic, torch-free hashed bag-of-words embedder —
instant, offline, and reproducible (so tests/CI never download a model). Set
``KF_EMBEDDING_BACKEND=sentence-transformers`` (and install the ``embeddings``
extra) to use all-MiniLM-L6-v2 instead. Both backends emit 384-dim vectors, so
the LanceDB schema is identical regardless of choice.
"""
from __future__ import annotations

import math
import os
import re

DIM = 384  # matches all-MiniLM-L6-v2 output dimension

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ST_MODEL = None  # lazily loaded sentence-transformers model


def _hashed_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        h = 0
        for ch in tok:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def _backend() -> str:
    return os.environ.get("KF_EMBEDDING_BACKEND", "lightweight").lower()


def _load_st():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer  # lazy, optional dep

        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts into 384-dim vectors using the active backend."""
    if not texts:
        return []
    if _backend() in ("sentence-transformers", "st", "sentence_transformers"):
        model = _load_st()
        arr = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return [list(map(float, row)) for row in arr]
    return [_hashed_embed(t) for t in texts]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
