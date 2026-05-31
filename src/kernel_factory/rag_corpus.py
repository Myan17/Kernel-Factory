"""
Declarative configuration of all source repositories and file patterns
for the production RAG corpus. Edit this file to add new sources.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RepoSource:
    name: str               # human label, e.g. "jax-ml/jax"
    url: str                # git clone URL
    branch: str             # branch to checkout
    globs: list[str]        # file glob patterns relative to repo root
    tier: int               # 1 (primary) to 4 (docs)
    description: str = ""


CORPUS_SOURCES: list[RepoSource] = [
    RepoSource(
        name="jax-ml/jax",
        url="https://github.com/jax-ml/jax",
        branch="main",
        globs=[
            "jax/experimental/pallas/ops/tpu/**/*.py",
        ],
        tier=1,
        description="Official JAX Pallas TPU kernels — flash attention, paged attention, megablox GMM",
    ),
    RepoSource(
        name="AI-Hypercomputer/maxtext",
        url="https://github.com/AI-Hypercomputer/maxtext",
        branch="main",
        globs=[
            "MaxText/layers/*.py",
            "MaxText/kernels/**/*.py",
        ],
        tier=2,
        description="MaxText production LLM training layers — RMSNorm, GQA, MoE linears",
    ),
    RepoSource(
        name="google-deepmind/recurrentgemma",
        url="https://github.com/google-deepmind/recurrentgemma",
        branch="main",
        globs=[
            "recurrentgemma/jax/pallas.py",
        ],
        tier=3,
        description="RecurrentGemma Griffin linear recurrence Pallas kernel",
    ),
]

# Files to exclude even if they match a glob (test files, init files, etc.)
EXCLUDE_PATTERNS: list[str] = [
    "**/test_*.py",
    "**/*_test.py",
    "**/__init__.py",
    "**/__pycache__/**",
    "**/conftest.py",
]

# Minimum lines for a file to be worth processing
MIN_FILE_LINES = 20
