"""LanceDB-backed template retrieval.

Uses a lightweight deterministic hashed bag-of-words embedding (stdlib only) —
no torch, no model download, fully offline. The TemplateRAG / LanceDB interface
is identical to a transformer-backed implementation; only ``_embed`` differs, so
swapping in sentence-transformers later is a localized change.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import lancedb

from kernel_factory.schemas import LayerSpec
from kernel_factory.templates import TEMPLATES

TABLE_NAME = "kernel_templates"
DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vec = [0.0] * DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        # Stable per-token bucket (Python's hash() is salted; use a fixed hash).
        h = 0
        for ch in tok:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def _template_label(op_type: str) -> str:
    return {
        "matmul": "MatMul BF16->F32",
        "rmsnorm": "RMSNorm BF16->F32",
    }.get(op_type, f"{op_type} template")


class TemplateRAG:
    def __init__(self, db_path: Path = Path(".lancedb")):
        self.db_path = Path(db_path)

    def _connect(self):
        return lancedb.connect(str(self.db_path))

    def seed(self, force: bool = False) -> int:
        """Embed and insert all templates from templates.TEMPLATES.
        Returns number of records inserted; 0 if already seeded and not forced."""
        db = self._connect()
        names = db.list_tables().tables
        if TABLE_NAME in names:
            if not force:
                return 0
            db.drop_table(TABLE_NAME)

        rows = [
            {
                "op_type": op_type,
                "template_name": _template_label(op_type),
                "template_code": code,
                "vector": _embed(code),
            }
            for op_type, code in TEMPLATES.items()
        ]
        db.create_table(TABLE_NAME, data=rows)
        return len(rows)

    def is_seeded(self) -> bool:
        db = self._connect()
        if TABLE_NAME not in db.list_tables().tables:
            return False
        return db.open_table(TABLE_NAME).count_rows() > 0

    def retrieve(self, spec: LayerSpec, top_k: int = 1) -> str:
        if not self.is_seeded():
            raise RuntimeError("RAG not seeded. Run seed() first.")
        db = self._connect()
        tbl = db.open_table(TABLE_NAME)
        query = (
            f"{spec.op_type} {spec.input_dtype.value} to {spec.output_dtype.value} "
            f"M={spec.M} N={spec.N} K={spec.K}"
        )
        hits = tbl.search(_embed(query)).limit(top_k).to_list()
        if not hits:
            raise RuntimeError("RAG returned no results.")
        return hits[0]["template_code"]
