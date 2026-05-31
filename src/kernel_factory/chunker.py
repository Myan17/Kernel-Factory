"""AST-based semantic chunker for Pallas kernel source files.

Splits Python source on complete function/class boundaries (never fixed token
windows), attaches kernel-class / op-type / tag metadata, and produces
deterministic chunk ids. Uses the stdlib ``ast`` module — no tree-sitter.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Spec says "discard under 5 lines"; its own test keeps a 4-line kernel and drops
# a 2-line stub, so the effective floor is 4 (keep >= 4, drop <= 3).
MIN_CHUNK_LINES = 4
MAX_CHUNK_LINES = 150

# kernel_class -> keyword fragments that imply it (checked against name + body)
KERNEL_CLASS_MAP: dict[str, list[str]] = {
    "matmul": ["matmul", "gemm", "dot", "linear", "gmm", "grouped_matmul"],
    "attention": ["attention", "flash_attn", "splash", "paged", "mha", "mqa", "gqa"],
    "norm": ["rmsnorm", "layernorm", "rms_norm", "layer_norm", "normalize"],
    "elementwise": ["add", "mul", "gelu", "silu", "relu", "softmax", "sigmoid"],
    "collective": ["all_gather", "reduce_scatter", "all_reduce", "alltoall"],
    "moe": ["moe", "mixture_of_experts", "routing", "expert"],
    "recurrence": ["scan", "recurrence", "rnn", "ssm", "griffin"],
}

# op_type normalization per kernel_class
_OP_TYPE = {
    "matmul": "matmul",
    "attention": "attention",
    "norm": "norm",
    "elementwise": "elementwise",
    "collective": "collective",
    "moe": "moe",
    "recurrence": "recurrence",
}

TAG_RULES: dict[str, list[str]] = {
    "bfloat16": ["bfloat16", "bf16"],
    "float32": ["float32", "f32"],
    "int8": ["int8"],
    "vmem": ["pltpu.VMEM", "VMEM("],
    "pipeline": ["async_copy", "get_pipelined", "double_buffer", "stages"],
    "flash": ["flash", "online_softmax"],
    "causal": ["causal", "mask", "tril"],
    "sparse": ["sparse", "block_sparse", "sparsity"],
    "quantized": ["quantize", "int8", "scale"],
    "pipelined": ["get_pipelined", "async_copy"],
    "multi_chip": ["shard_map", "all_gather", "reduce_scatter"],
}


@dataclass
class KernelChunk:
    chunk_id: str
    source_repo: str
    source_file: str
    function_name: str
    kernel_class: str
    op_type: str
    tags: list[str]
    chunk_text: str
    tier: int
    line_start: int
    line_end: int
    vector: list[float] = field(default_factory=list)


class KernelChunker:
    def chunk_file(
        self,
        source_code: str,
        source_file: str,
        source_repo: str,
        tier: int,
    ) -> list[KernelChunk]:
        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            log.warning("Skipping %s — SyntaxError: %s", source_file, exc)
            return []

        lines = source_code.splitlines()
        chunks: list[KernelChunk] = []

        # Only top-level functions/classes (direct children of the module).
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue

            start = self._node_start(node)
            end = node.end_lineno or start
            block = lines[start - 1 : end]
            if len(block) < MIN_CHUNK_LINES:
                continue

            sub_blocks = self._split_if_large(block)
            for text, off in sub_blocks:
                chunks.append(
                    self._build_chunk(
                        text=text,
                        name=node.name,
                        source_file=source_file,
                        source_repo=source_repo,
                        tier=tier,
                        line_start=start + off,
                        line_end=start + off + len(text.splitlines()) - 1,
                    )
                )
        return chunks

    @staticmethod
    def _node_start(node) -> int:
        # Include any decorators above the def/class.
        decos = getattr(node, "decorator_list", [])
        if decos:
            return min(d.lineno for d in decos)
        return node.lineno

    def _split_if_large(self, block: list[str]) -> list[tuple[str, int]]:
        """Return list of (text, line_offset) sub-blocks.

        A block <= MAX_CHUNK_LINES is returned whole. Larger blocks are split at
        '# ───' separator lines, with the signature (first 3 lines) prepended to
        each sub-block so every chunk is self-contained.
        """
        if len(block) <= MAX_CHUNK_LINES:
            return [("\n".join(block), 0)]

        signature = block[:3]
        sep_idxs = [
            i for i, ln in enumerate(block)
            if i > 2 and ln.lstrip().startswith("#") and "─" in ln
        ]
        if not sep_idxs:
            return [("\n".join(block), 0)]

        sub_blocks: list[tuple[str, int]] = []
        bounds = [3] + sep_idxs + [len(block)]
        for a, b in zip(bounds, bounds[1:]):
            piece = block[a:b]
            if len(piece) < MIN_CHUNK_LINES:
                continue
            text = "\n".join(signature + piece)
            sub_blocks.append((text, a))
        # Fall back to the whole block if separators produced nothing usable.
        return sub_blocks or [("\n".join(block), 0)]

    def _build_chunk(
        self,
        text: str,
        name: str,
        source_file: str,
        source_repo: str,
        tier: int,
        line_start: int,
        line_end: int,
    ) -> KernelChunk:
        kernel_class, op_type = self._infer_kernel_class(name, text)
        return KernelChunk(
            chunk_id=self._make_chunk_id(source_file, name, text),
            source_repo=source_repo,
            source_file=source_file,
            function_name=name,
            kernel_class=kernel_class,
            op_type=op_type,
            tags=self._extract_tags(name, text),
            chunk_text=text,
            tier=tier,
            line_start=line_start,
            line_end=line_end,
        )

    def _infer_kernel_class(self, name: str, body: str) -> tuple[str, str]:
        haystack = f"{name}\n{body}".lower()
        name_lower = name.lower()
        # Prefer a match in the function name; fall back to the body.
        for source in (name_lower, haystack):
            for kclass, keywords in KERNEL_CLASS_MAP.items():
                if any(kw in source for kw in keywords):
                    return kclass, _OP_TYPE[kclass]
        return "unknown", "unknown"

    def _extract_tags(self, name: str, body: str) -> list[str]:
        haystack = f"{name}\n{body}".lower()
        tags = []
        for tag, keywords in TAG_RULES.items():
            if any(kw.lower() in haystack for kw in keywords):
                tags.append(tag)
        return tags

    def _make_chunk_id(self, source_file: str, function_name: str, text: str) -> str:
        digest = hashlib.sha256(
            f"{source_file}|{function_name}|{text}".encode("utf-8")
        ).hexdigest()
        return digest[:12]
