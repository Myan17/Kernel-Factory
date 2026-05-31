"""LanceDB-backed template/kernel retrieval.

Two classes live here:

- ``ProductionRAG`` — the production retriever over a rich corpus of real Pallas
  kernels (populated by ``scripts/ingest_rag.py``). Rich schema, dedup, metadata
  filtering, stats, and a static-template fallback when the corpus is empty.
- ``TemplateRAG`` — the original lightweight 2-template store, kept as a legacy
  shim so the simple ``seed`` flow and existing callers keep working.

Embeddings come from ``kernel_factory.embeddings`` (torch-free by default,
sentence-transformers opt-in). Both backends emit 384-dim vectors.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import lancedb
import pyarrow as pa

from kernel_factory.embeddings import DIM, embed_one, embed_texts
from kernel_factory.schemas import LayerSpec
from kernel_factory.templates import TEMPLATES

# ── Legacy TemplateRAG (2-template store) ─────────────────────────────────────

TABLE_NAME = "kernel_templates"
_LEGACY_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding for the legacy store."""
    vec = [0.0] * _LEGACY_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        h = 0
        for ch in tok:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        vec[h % _LEGACY_DIM] += 1.0
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

    def retrieve_one(self, spec: LayerSpec) -> str:
        """Alias for retrieve(); shared interface with ProductionRAG."""
        return self.retrieve(spec)


# ── ProductionRAG (rich corpus) ───────────────────────────────────────────────

PROD_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_repo", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("function_name", pa.string()),
    pa.field("kernel_class", pa.string()),
    pa.field("op_type", pa.string()),
    pa.field("tags", pa.list_(pa.string())),
    pa.field("chunk_text", pa.string()),
    pa.field("tier", pa.int32()),
    pa.field("line_start", pa.int32()),
    pa.field("line_end", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

_RETURN_FIELDS = [
    "chunk_id", "source_repo", "source_file", "function_name",
    "kernel_class", "op_type", "tags", "chunk_text", "tier",
    "line_start", "line_end",
]

# kernel_class implied by op_type for the static fallback result
_OP_TO_CLASS = {"matmul": "matmul", "rmsnorm": "norm"}


def _embedding_input(chunk) -> str:
    """Metadata-enriched text used for the vector (not the stored chunk_text).

    Prepending op_type / kernel_class / function_name / tags gives the
    discriminative signal (e.g. 'matmul' vs 'norm') enough weight that retrieval
    isn't drowned out by shared Pallas boilerplate tokens.
    """
    meta = f"{chunk.op_type} {chunk.kernel_class} {chunk.function_name} {' '.join(chunk.tags)}"
    return f"{meta}\n{chunk.chunk_text}"


def _build_query(spec: LayerSpec) -> str:
    parts = [
        spec.op_type,
        spec.input_dtype.value,
        f"to {spec.output_dtype.value}",
        f"M={spec.M} N={spec.N}",
        "pallas_call BlockSpec",
        "tpu kernel",
    ]
    return " ".join(parts)


class ProductionRAG:
    TABLE_NAME = "kernel_chunks"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MIN_SEEDED = 3

    def __init__(self, db_path: Path = Path(".lancedb")):
        self.db_path = Path(db_path)

    def _connect(self):
        return lancedb.connect(str(self.db_path))

    def _table(self):
        db = self._connect()
        if self.TABLE_NAME not in db.list_tables().tables:
            return None
        return db.open_table(self.TABLE_NAME)

    def _scan_all(self) -> list[dict]:
        tbl = self._table()
        if tbl is None or tbl.count_rows() == 0:
            return []
        # Full scan via zero-vector search with a large limit (no pylance needed).
        return tbl.search([0.0] * DIM).limit(10_000_000).to_list()

    def count(self) -> int:
        tbl = self._table()
        return tbl.count_rows() if tbl is not None else 0

    def is_seeded(self) -> bool:
        return self.count() >= self.MIN_SEEDED

    def upsert_chunks(self, chunks: list) -> int:
        """Embed and upsert chunks. Dedups on chunk_id. Returns count inserted."""
        if not chunks:
            return 0
        db = self._connect()
        existing = {r["chunk_id"] for r in self._scan_all()}

        fresh = []
        seen_in_batch: set[str] = set()
        for c in chunks:
            if c.chunk_id in existing or c.chunk_id in seen_in_batch:
                continue
            seen_in_batch.add(c.chunk_id)
            fresh.append(c)
        if not fresh:
            return 0

        vectors = embed_texts([_embedding_input(c) for c in fresh])
        rows = [
            {
                "chunk_id": c.chunk_id,
                "source_repo": c.source_repo,
                "source_file": c.source_file,
                "function_name": c.function_name,
                "kernel_class": c.kernel_class,
                "op_type": c.op_type,
                "tags": list(c.tags),
                "chunk_text": c.chunk_text,
                "tier": int(c.tier),
                "line_start": int(c.line_start),
                "line_end": int(c.line_end),
                "vector": vec,
            }
            for c, vec in zip(fresh, vectors)
        ]

        if self.TABLE_NAME in db.list_tables().tables:
            db.open_table(self.TABLE_NAME).add(rows)
        else:
            db.create_table(self.TABLE_NAME, data=rows, schema=PROD_SCHEMA)
        return len(rows)

    def retrieve(
        self,
        spec: LayerSpec,
        top_k: int = 3,
        kernel_class_filter: str | None = None,
    ) -> list[dict]:
        if not self.is_seeded():
            return [self._fallback_result(spec)]
        tbl = self._table()
        search = tbl.search(embed_one(_build_query(spec)))
        if kernel_class_filter is not None:
            search = search.where(
                f"kernel_class = '{kernel_class_filter}'", prefilter=True
            )
        hits = search.limit(top_k).to_list()
        return [self._clean(h) for h in hits]

    def retrieve_one(self, spec: LayerSpec) -> str:
        results = self.retrieve(spec, top_k=1)
        if results:
            return results[0]["chunk_text"]
        return self._fallback_result(spec)["chunk_text"]

    def stats(self) -> dict:
        rows = self._scan_all()
        return {
            "total_chunks": len(rows),
            "by_kernel_class": dict(Counter(r["kernel_class"] for r in rows)),
            "by_source_repo": dict(Counter(r["source_repo"] for r in rows)),
            "by_tier": dict(Counter(int(r["tier"]) for r in rows)),
        }

    @staticmethod
    def _clean(row: dict) -> dict:
        return {k: row[k] for k in _RETURN_FIELDS if k in row}

    @staticmethod
    def _fallback_result(spec: LayerSpec) -> dict:
        code = TEMPLATES.get(spec.op_type) or next(iter(TEMPLATES.values()))
        return {
            "chunk_id": f"static_{spec.op_type}",
            "source_repo": "static",
            "source_file": "templates.py",
            "function_name": f"run_{spec.op_type}",
            "kernel_class": _OP_TO_CLASS.get(spec.op_type, "unknown"),
            "op_type": spec.op_type,
            "tags": [],
            "chunk_text": code,
            "tier": 0,
            "line_start": 0,
            "line_end": 0,
        }
