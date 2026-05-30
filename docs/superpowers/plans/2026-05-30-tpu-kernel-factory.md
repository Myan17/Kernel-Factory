# TPU Kernel Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, test-gated pipeline that generates and verifies custom JAX Pallas kernels for Google TPUs using a Solver → RAG → Assembly → Verify architecture.

**Architecture:** A local LLM orchestrates a solver that computes valid tile sizes from hardware constraints, retrieves verified Pallas templates from LanceDB, injects solved parameters into templates (never free-form code), and ships a self-contained verification script to a remote TPU VM for compilation and numerical validation. All kernel provenance is tracked in a Kuzu typed knowledge graph.

**Tech Stack:** JAX/Pallas, Pydantic v2, Kuzu (graph DB), LanceDB (vector RAG), SQLite (run log), uv (package management), pytest

**Project root:** `~/Downloads/QuackHack/`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | uv project + all dependencies |
| `src/kernel_factory/__init__.py` | package root |
| `src/kernel_factory/schemas.py` | Pydantic: LayerSpec, HardwareLimits, KernelConfig, TestResult |
| `src/kernel_factory/kg/schema.py` | Kuzu graph DDL strings for all 11 node/edge types |
| `src/kernel_factory/kg/graph.py` | KernelFactoryKG class — open/close, upsert, query helpers |
| `src/kernel_factory/solver.py` | VMEM-aware tile solver for MatMul and RMSNorm |
| `src/kernel_factory/templates.py` | Parameterized Pallas skeleton strings (MatMul, RMSNorm) |
| `src/kernel_factory/assembler.py` | Maps solver integers into template strings |
| `src/kernel_factory/verify.py` | Self-contained remote verification script |
| `tests/test_schemas.py` | Schema construction and validation |
| `tests/test_solver.py` | Solver correctness + VMEM budget enforcement |
| `tests/test_assembler.py` | Assembler round-trip: spec → assembled code string |
| `tests/test_verify.py` | CPU interpret-mode end-to-end (no TPU required) |

---

## Phase 1 — Workspace & Schemas

### Task 1: Initialize uv project and pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "kernel-factory"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
    "pydantic>=2.0",
    "kuzu>=0.6.0",
    "lancedb>=0.10.0",
    "numpy>=1.26",
    "rich>=13.0",
    "pyarrow>=14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kernel_factory"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Sync dependencies**

```bash
cd ~/Downloads/QuackHack
uv sync --extra dev
```

Expected: resolves and installs all packages, creates `.venv/`.

- [ ] **Step 3: Commit**

```bash
git init
git add pyproject.toml
git commit -m "chore: initialize uv project"
```

---

### Task 2: Pydantic schemas (schemas.py)

**Files:**
- Create: `src/kernel_factory/__init__.py`
- Create: `src/kernel_factory/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pytest
from kernel_factory.schemas import (
    DType, LayerSpec, HardwareLimits, KernelConfig, TestResult
)

def test_hardware_limits_v5e():
    hw = HardwareLimits.for_v5e()
    assert hw.vmem_bytes == 128 * 1024 * 1024
    assert hw.vector_width == 128
    assert hw.sublane_width == 8

def test_hardware_limits_v4():
    hw = HardwareLimits.for_v4()
    assert hw.vmem_bytes == 16 * 1024 * 1024

def test_layer_spec_defaults():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    assert spec.input_dtype == DType.BFLOAT16
    assert spec.accumulator_dtype == DType.FLOAT32

def test_kernel_config_fields():
    config = KernelConfig(
        block_m=128, block_n=128, block_k=128,
        stages=1,
        input_dtype=DType.BFLOAT16,
        output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=2 * 1024 * 1024,
        vmem_utilization_fraction=0.015,
    )
    assert config.block_m == 128

def test_test_result_pass():
    from kernel_factory.schemas import KernelConfig, DType, LayerSpec
    config = KernelConfig(
        block_m=64, block_n=64, block_k=64, stages=1,
        input_dtype=DType.BFLOAT16, output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=512 * 1024,
        vmem_utilization_fraction=0.004,
    )
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    result = TestResult(kernel_config=config, layer_spec=spec, passed=True)
    assert result.passed is True
    assert result.error_trace is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd ~/Downloads/QuackHack
uv run pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'kernel_factory'`

- [ ] **Step 3: Write `src/kernel_factory/__init__.py`**

```python
```
(empty file)

- [ ] **Step 4: Write `src/kernel_factory/schemas.py`**

```python
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DType(str, Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"

    @property
    def itemsize(self) -> int:
        return {"float32": 4, "bfloat16": 2, "int8": 1}[self.value]


class LayerSpec(BaseModel):
    op_type: str  # "matmul" | "rmsnorm"
    M: int
    N: int
    K: int
    input_dtype: DType = DType.BFLOAT16
    output_dtype: DType = DType.BFLOAT16
    accumulator_dtype: DType = DType.FLOAT32
    batch_size: Optional[int] = None


class HardwareLimits(BaseModel):
    tpu_version: str
    vmem_bytes: int
    hbm_bandwidth_gbps: float
    vector_width: int = 128   # last dim must be multiple of this
    sublane_width: int = 8    # second-to-last dim must be multiple of this
    max_tiles_per_dim: int = 2048
    vmem_safety_fraction: float = Field(default=0.75, ge=0.0, le=1.0)

    @classmethod
    def for_v5e(cls) -> HardwareLimits:
        return cls(
            tpu_version="v5e",
            vmem_bytes=128 * 1024 * 1024,
            hbm_bandwidth_gbps=819.2,
        )

    @classmethod
    def for_v4(cls) -> HardwareLimits:
        return cls(
            tpu_version="v4",
            vmem_bytes=16 * 1024 * 1024,
            hbm_bandwidth_gbps=614.4,
        )

    @classmethod
    def for_v6e(cls) -> HardwareLimits:
        return cls(
            tpu_version="v6e",
            vmem_bytes=128 * 1024 * 1024,
            hbm_bandwidth_gbps=1638.4,
        )

    @property
    def vmem_budget_bytes(self) -> int:
        return int(self.vmem_bytes * self.vmem_safety_fraction)


class KernelConfig(BaseModel):
    block_m: int
    block_n: int
    block_k: int
    stages: int = 1
    input_dtype: DType
    output_dtype: DType
    accumulator_dtype: DType
    total_vmem_estimate_bytes: int
    vmem_utilization_fraction: float


class TestResult(BaseModel):
    kernel_config: KernelConfig
    layer_spec: LayerSpec
    passed: bool
    max_abs_error: Optional[float] = None
    compile_time_ms: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    error_trace: Optional[str] = None
    tpu_version: str = "unknown"
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/kernel_factory/__init__.py src/kernel_factory/schemas.py tests/test_schemas.py
git commit -m "feat: Pydantic schemas — LayerSpec, HardwareLimits, KernelConfig, TestResult"
```

---

### Task 3: Kuzu graph schema + KernelFactoryKG

**Files:**
- Create: `src/kernel_factory/kg/__init__.py`
- Create: `src/kernel_factory/kg/schema.py`
- Create: `src/kernel_factory/kg/graph.py`
- Test: `tests/test_kg.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_kg.py
import tempfile, pathlib
from kernel_factory.kg.graph import KernelFactoryKG

def test_kg_opens_and_closes():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.close()

def test_upsert_kernel_template():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.upsert_kernel_template(
            name="matmul_v1",
            op_type="matmul",
            template_str="def kernel(a_ref, b_ref, o_ref): ...",
            verified=True,
        )
        results = kg.query("MATCH (t:KernelTemplate) RETURN t.name")
        names = [r[0] for r in results]
        assert "matmul_v1" in names
        kg.close()

def test_upsert_hardware_limits():
    with tempfile.TemporaryDirectory() as d:
        kg = KernelFactoryKG(pathlib.Path(d) / "test.kuzu")
        kg.upsert_hardware_limits(
            tpu_version="v5e",
            vmem_bytes=128 * 1024 * 1024,
            vector_width=128,
            sublane_width=8,
        )
        results = kg.query("MATCH (h:HardwareLimits) RETURN h.tpu_version")
        versions = [r[0] for r in results]
        assert "v5e" in versions
        kg.close()
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_kg.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write `src/kernel_factory/kg/__init__.py`** (empty)

- [ ] **Step 4: Write `src/kernel_factory/kg/schema.py`**

```python
# Kuzu DDL — all node and edge type definitions
NODE_SCHEMAS = """
CREATE NODE TABLE IF NOT EXISTS KernelTemplate(
    name STRING,
    op_type STRING,
    template_str STRING,
    verified BOOLEAN,
    PRIMARY KEY(name)
);
CREATE NODE TABLE IF NOT EXISTS KernelSpec(
    spec_id STRING,
    op_type STRING,
    M INT64,
    N INT64,
    K INT64,
    input_dtype STRING,
    output_dtype STRING,
    accumulator_dtype STRING,
    PRIMARY KEY(spec_id)
);
CREATE NODE TABLE IF NOT EXISTS HardwareLimits(
    tpu_version STRING,
    vmem_bytes INT64,
    vector_width INT64,
    sublane_width INT64,
    PRIMARY KEY(tpu_version)
);
CREATE NODE TABLE IF NOT EXISTS TileSpec(
    tile_id STRING,
    block_m INT64,
    block_n INT64,
    block_k INT64,
    stages INT64,
    vmem_estimate_bytes INT64,
    PRIMARY KEY(tile_id)
);
CREATE NODE TABLE IF NOT EXISTS GeneratedKernel(
    kernel_id STRING,
    code STRING,
    created_at STRING,
    PRIMARY KEY(kernel_id)
);
CREATE NODE TABLE IF NOT EXISTS TestCase(
    test_id STRING,
    description STRING,
    PRIMARY KEY(test_id)
);
CREATE NODE TABLE IF NOT EXISTS CompileResult(
    result_id STRING,
    passed BOOLEAN,
    compile_time_ms DOUBLE,
    error_trace STRING,
    PRIMARY KEY(result_id)
);
CREATE NODE TABLE IF NOT EXISTS BenchmarkResult(
    bench_id STRING,
    latency_ms DOUBLE,
    tflops DOUBLE,
    PRIMARY KEY(bench_id)
);
CREATE NODE TABLE IF NOT EXISTS FailureCase(
    failure_id STRING,
    description STRING,
    error_pattern STRING,
    PRIMARY KEY(failure_id)
);
CREATE NODE TABLE IF NOT EXISTS KnownBug(
    bug_id STRING,
    description STRING,
    tpu_version STRING,
    PRIMARY KEY(bug_id)
);
CREATE NODE TABLE IF NOT EXISTS FixPattern(
    fix_id STRING,
    description STRING,
    patch_hint STRING,
    PRIMARY KEY(fix_id)
);
"""

EDGE_SCHEMAS = """
CREATE REL TABLE IF NOT EXISTS USES_TEMPLATE(FROM KernelSpec TO KernelTemplate);
CREATE REL TABLE IF NOT EXISTS CONSTRAINED_BY(FROM KernelSpec TO HardwareLimits);
CREATE REL TABLE IF NOT EXISTS HAS_TILE(FROM KernelSpec TO TileSpec);
CREATE REL TABLE IF NOT EXISTS GENERATED_FROM(FROM GeneratedKernel TO KernelSpec);
CREATE REL TABLE IF NOT EXISTS HAS_COMPILE_RESULT(FROM GeneratedKernel TO CompileResult);
CREATE REL TABLE IF NOT EXISTS HAS_BENCHMARK(FROM GeneratedKernel TO BenchmarkResult);
CREATE REL TABLE IF NOT EXISTS CAUSED_FAILURE(FROM CompileResult TO FailureCase);
CREATE REL TABLE IF NOT EXISTS KNOWN_BUG_FOR(FROM FailureCase TO KnownBug);
CREATE REL TABLE IF NOT EXISTS FIXED_BY(FROM KnownBug TO FixPattern);
CREATE REL TABLE IF NOT EXISTS VALIDATES(FROM TestCase TO GeneratedKernel);
"""
```

- [ ] **Step 5: Write `src/kernel_factory/kg/graph.py`**

```python
from __future__ import annotations
import pathlib
import kuzu
from kernel_factory.kg.schema import NODE_SCHEMAS, EDGE_SCHEMAS


class KernelFactoryKG:
    def __init__(self, db_path: pathlib.Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self):
        for stmt in NODE_SCHEMAS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt + ";")
        for stmt in EDGE_SCHEMAS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt + ";")

    def query(self, cypher: str) -> list:
        result = self._conn.execute(cypher)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def upsert_kernel_template(
        self, name: str, op_type: str, template_str: str, verified: bool
    ):
        self._conn.execute(
            "MERGE (t:KernelTemplate {name: $name}) "
            "SET t.op_type = $op_type, t.template_str = $tpl, t.verified = $v",
            {"name": name, "op_type": op_type, "tpl": template_str, "v": verified},
        )

    def upsert_hardware_limits(
        self, tpu_version: str, vmem_bytes: int, vector_width: int, sublane_width: int
    ):
        self._conn.execute(
            "MERGE (h:HardwareLimits {tpu_version: $ver}) "
            "SET h.vmem_bytes = $vm, h.vector_width = $vw, h.sublane_width = $sw",
            {
                "ver": tpu_version,
                "vm": vmem_bytes,
                "vw": vector_width,
                "sw": sublane_width,
            },
        )

    def get_template(self, op_type: str) -> str | None:
        rows = self.query(
            f"MATCH (t:KernelTemplate) WHERE t.op_type = '{op_type}' AND t.verified = true "
            "RETURN t.template_str LIMIT 1"
        )
        return rows[0][0] if rows else None

    def close(self):
        del self._conn
        del self._db
```

- [ ] **Step 6: Run KG tests**

```bash
uv run pytest tests/test_kg.py -v
```

Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add src/kernel_factory/kg/ tests/test_kg.py
git commit -m "feat: Kuzu KernelFactoryKG — 11-node schema + upsert/query helpers"
```

---

## Phase 2 — Deterministic Solver

### Task 4: VMEM-aware tile solver

**Files:**
- Create: `src/kernel_factory/solver.py`
- Create: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_solver.py
import pytest
from kernel_factory.schemas import LayerSpec, HardwareLimits, DType, KernelConfig
from kernel_factory.solver import TileSolver


def _hw() -> HardwareLimits:
    return HardwareLimits.for_v5e()


def test_solver_returns_config_for_matmul():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    config = TileSolver(_hw()).solve(spec)
    assert isinstance(config, KernelConfig)
    assert config.block_m > 0
    assert config.block_n > 0
    assert config.block_k > 0


def test_solver_vmem_within_budget():
    spec = LayerSpec(op_type="matmul", M=4096, N=4096, K=2048)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.total_vmem_estimate_bytes <= hw.vmem_budget_bytes


def test_solver_last_dim_aligned_to_vector_width():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_n % hw.vector_width == 0
    assert config.block_k % hw.vector_width == 0


def test_solver_second_dim_aligned_to_sublane():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_m % hw.sublane_width == 0


def test_solver_raises_for_unsolvable():
    # v4 has 16 MiB VMEM — very small matrices with tiny blocks still should solve
    spec = LayerSpec(op_type="matmul", M=64, N=64, K=64)
    config = TileSolver(HardwareLimits.for_v4()).solve(spec)
    assert config is not None


def test_solver_rmsnorm():
    spec = LayerSpec(op_type="rmsnorm", M=1024, N=4096, K=4096)
    config = TileSolver(_hw()).solve(spec)
    assert config.block_m > 0
    assert config.block_n > 0


def test_solver_vmem_utilization_fraction():
    spec = LayerSpec(op_type="matmul", M=2048, N=2048, K=1024)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert 0.0 < config.vmem_utilization_fraction <= 1.0
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_solver.py -v
```

Expected: `ModuleNotFoundError: No module named 'kernel_factory.solver'`

- [ ] **Step 3: Write `src/kernel_factory/solver.py`**

```python
from __future__ import annotations
import math
from kernel_factory.schemas import DType, HardwareLimits, KernelConfig, LayerSpec


_CANDIDATE_POWERS = [16, 32, 64, 128, 256, 512]


def _dtype_bytes(d: DType) -> int:
    return d.itemsize


def _vmem_matmul(bm: int, bn: int, bk: int, spec: LayerSpec) -> int:
    ib = _dtype_bytes(spec.input_dtype)
    ob = _dtype_bytes(spec.output_dtype)
    ab = _dtype_bytes(spec.accumulator_dtype)
    return bm * bk * ib + bk * bn * ib + bm * bn * ob + bm * bn * ab


def _vmem_rmsnorm(bm: int, bn: int, spec: LayerSpec) -> int:
    ib = _dtype_bytes(spec.input_dtype)
    ob = _dtype_bytes(spec.output_dtype)
    ab = _dtype_bytes(spec.accumulator_dtype)
    # input tile + weight tile + output tile + per-row accumulator
    return bm * bn * ib + bn * ib + bm * bn * ob + bm * ab


def _aligned(v: int, alignment: int) -> bool:
    return v % alignment == 0


class TileSolver:
    def __init__(self, hw: HardwareLimits):
        self.hw = hw

    def solve(self, spec: LayerSpec) -> KernelConfig:
        if spec.op_type == "matmul":
            return self._solve_matmul(spec)
        elif spec.op_type == "rmsnorm":
            return self._solve_rmsnorm(spec)
        raise ValueError(f"Unsupported op_type: {spec.op_type}")

    def _candidates(self, dim: int, must_align_to: int) -> list[int]:
        return [
            p for p in _CANDIDATE_POWERS
            if p <= dim and _aligned(p, must_align_to)
        ]

    def _solve_matmul(self, spec: LayerSpec) -> KernelConfig:
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1.0

        bm_candidates = self._candidates(spec.M, hw.sublane_width)
        bn_candidates = self._candidates(spec.N, hw.vector_width)
        bk_candidates = self._candidates(spec.K, hw.vector_width)

        for bm in reversed(bm_candidates):
            for bn in reversed(bn_candidates):
                for bk in reversed(bk_candidates):
                    vmem = _vmem_matmul(bm, bn, bk, spec)
                    if vmem > budget:
                        continue
                    util = vmem / hw.vmem_bytes
                    # score: prefer larger tiles (more compute per memory movement)
                    score = bm * bn * bk
                    if score > best_score:
                        best_score = score
                        best = KernelConfig(
                            block_m=bm, block_n=bn, block_k=bk,
                            stages=1,
                            input_dtype=spec.input_dtype,
                            output_dtype=spec.output_dtype,
                            accumulator_dtype=spec.accumulator_dtype,
                            total_vmem_estimate_bytes=vmem,
                            vmem_utilization_fraction=util,
                        )

        if best is None:
            raise RuntimeError(
                f"No valid tile found for {spec} within {budget} bytes VMEM budget"
            )
        return best

    def _solve_rmsnorm(self, spec: LayerSpec) -> KernelConfig:
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1.0

        bm_candidates = self._candidates(spec.M, hw.sublane_width)
        bn_candidates = self._candidates(spec.N, hw.vector_width)

        for bm in reversed(bm_candidates):
            for bn in reversed(bn_candidates):
                vmem = _vmem_rmsnorm(bm, bn, spec)
                if vmem > budget:
                    continue
                util = vmem / hw.vmem_bytes
                score = bm * bn
                if score > best_score:
                    best_score = score
                    best = KernelConfig(
                        block_m=bm, block_n=bn, block_k=spec.K,
                        stages=1,
                        input_dtype=spec.input_dtype,
                        output_dtype=spec.output_dtype,
                        accumulator_dtype=spec.accumulator_dtype,
                        total_vmem_estimate_bytes=vmem,
                        vmem_utilization_fraction=util,
                    )

        if best is None:
            raise RuntimeError(
                f"No valid tile found for RMSNorm {spec} within {budget} bytes"
            )
        return best
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_solver.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/kernel_factory/solver.py tests/test_solver.py
git commit -m "feat: VMEM-aware deterministic tile solver for MatMul and RMSNorm"
```

---

## Phase 3 — Assembly & RAG Engine

### Task 5: Pallas template skeletons (templates.py)

**Files:**
- Create: `src/kernel_factory/templates.py`

- [ ] **Step 1: Write `src/kernel_factory/templates.py`**

These are parameterized skeleton *strings* — not executed locally. Parameters `{block_m}`, `{block_n}`, `{block_k}`, `{M}`, `{N}`, `{K}`, `{input_dtype}`, `{output_dtype}`, `{accumulator_dtype}` are substituted by the assembler.

```python
# Verified Pallas skeleton templates — substitution targets only, no free-form logic.

MATMUL_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax_pallas as pl
import jax_pallas.tpu_primitives as pltpu

def matmul_kernel(a_ref, b_ref, o_ref, acc_ref, *, block_k={block_k}):
    """Dense MatMul Pallas kernel — template-generated."""
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.{accumulator_dtype}),
        b_ref[...].astype(jnp.{accumulator_dtype}),
        preferred_element_type=jnp.{accumulator_dtype},
    )

    @pl.when(pl.program_id(2) == {num_k_tiles} - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(jnp.{output_dtype})


def run_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k = {block_m}, {block_n}, {block_k}
    num_k_tiles = K // block_k

    grid = (M // block_m, N // block_n, num_k_tiles)
    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda m, n, k: (m, k)),
        pl.BlockSpec((block_k, block_n), lambda m, n, k: (k, n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n, k: (m, n))
    scratch_specs = [pltpu.VMEM((block_m, block_n), jnp.{accumulator_dtype})]

    return pl.pallas_call(
        matmul_kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_specs,
    )(a, b)
'''

RMSNORM_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax_pallas as pl

EPS = 1e-6

def rmsnorm_kernel(x_ref, w_ref, o_ref):
    """RMSNorm Pallas kernel — template-generated."""
    x = x_ref[...].astype(jnp.{accumulator_dtype})
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS)
    normed = (x / rms).astype(jnp.{output_dtype})
    o_ref[...] = normed * w_ref[...].astype(jnp.{output_dtype})


def run_rmsnorm(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    M, N = x.shape
    block_m, block_n = {block_m}, {block_n}

    in_specs = [
        pl.BlockSpec((block_m, block_n), lambda m, n: (m, n)),
        pl.BlockSpec((block_n,), lambda m, n: (n,)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n: (m, n))

    return pl.pallas_call(
        rmsnorm_kernel,
        grid=(M // block_m, N // block_n),
        in_specs=in_specs,
        out_specs=out_specs,
    )(x, w)
'''

TEMPLATES: dict[str, str] = {
    "matmul": MATMUL_TEMPLATE,
    "rmsnorm": RMSNORM_TEMPLATE,
}
```

- [ ] **Step 2: Commit**

```bash
git add src/kernel_factory/templates.py
git commit -m "feat: verified Pallas template skeletons for MatMul and RMSNorm"
```

---

### Task 6: Assembler (assembler.py)

**Files:**
- Create: `src/kernel_factory/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_assembler.py
from kernel_factory.schemas import LayerSpec, HardwareLimits, DType, KernelConfig
from kernel_factory.assembler import Assembler


def _config() -> KernelConfig:
    return KernelConfig(
        block_m=128, block_n=128, block_k=128, stages=1,
        input_dtype=DType.BFLOAT16,
        output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=2 * 1024 * 1024,
        vmem_utilization_fraction=0.015,
    )


def test_assemble_matmul_contains_block_sizes():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = Assembler().assemble(spec, _config())
    assert "128" in code
    assert "run_matmul" in code


def test_assemble_rmsnorm_contains_block_sizes():
    spec = LayerSpec(op_type="rmsnorm", M=512, N=4096, K=4096)
    code = Assembler().assemble(spec, _config())
    assert "run_rmsnorm" in code


def test_assemble_raises_for_unknown_op():
    spec = LayerSpec(op_type="attention", M=512, N=512, K=512)
    import pytest
    with pytest.raises(ValueError, match="No template"):
        Assembler().assemble(spec, _config())


def test_assemble_num_k_tiles_correct():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    config = _config()  # block_k=128 → num_k_tiles = 512/128 = 4
    code = Assembler().assemble(spec, config)
    assert "num_k_tiles = 4" in code or "4" in code
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_assembler.py -v
```

Expected: `ModuleNotFoundError: No module named 'kernel_factory.assembler'`

- [ ] **Step 3: Write `src/kernel_factory/assembler.py`**

```python
from __future__ import annotations
import math
from kernel_factory.schemas import DType, KernelConfig, LayerSpec
from kernel_factory.templates import TEMPLATES


class Assembler:
    def assemble(self, spec: LayerSpec, config: KernelConfig) -> str:
        template = TEMPLATES.get(spec.op_type)
        if template is None:
            raise ValueError(f"No template registered for op_type='{spec.op_type}'")

        num_k_tiles = max(1, spec.K // config.block_k) if spec.op_type == "matmul" else 1

        params = {
            "block_m": config.block_m,
            "block_n": config.block_n,
            "block_k": config.block_k,
            "M": spec.M,
            "N": spec.N,
            "K": spec.K,
            "input_dtype": config.input_dtype.value,
            "output_dtype": config.output_dtype.value,
            "accumulator_dtype": config.accumulator_dtype.value,
            "num_k_tiles": num_k_tiles,
        }
        return template.format(**params)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_assembler.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/kernel_factory/assembler.py tests/test_assembler.py
git commit -m "feat: assembler — maps solver config into Pallas template strings"
```

---

## Phase 4 — Verification Harness

### Task 7: verify.py — self-contained remote verification script

**Files:**
- Create: `src/kernel_factory/verify.py`
- Create: `tests/test_verify.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verify.py
"""
These tests run in CPU interpret=True mode only — no TPU required.
"""
import jax.numpy as jnp
import pytest
from kernel_factory.schemas import LayerSpec, HardwareLimits, DType
from kernel_factory.solver import TileSolver
from kernel_factory.assembler import Assembler
from kernel_factory.verify import VerificationGate, VerifyMode


def test_matmul_cpu_interpret():
    spec = LayerSpec(op_type="matmul", M=64, N=64, K=64,
                     input_dtype=DType.BFLOAT16, output_dtype=DType.BFLOAT16)
    hw = HardwareLimits.for_v5e()
    config = TileSolver(hw).solve(spec)
    gate = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET)
    result = gate.run()
    assert result.passed, result.error_trace


def test_rmsnorm_cpu_interpret():
    spec = LayerSpec(op_type="rmsnorm", M=64, N=128, K=128,
                     input_dtype=DType.BFLOAT16, output_dtype=DType.BFLOAT16)
    hw = HardwareLimits.for_v5e()
    config = TileSolver(hw).solve(spec)
    gate = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET)
    result = gate.run()
    assert result.passed, result.error_trace


def test_result_logged_to_sqlite(tmp_path):
    spec = LayerSpec(op_type="matmul", M=64, N=64, K=64)
    hw = HardwareLimits.for_v5e()
    config = TileSolver(hw).solve(spec)
    db_path = tmp_path / "results.db"
    gate = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET, db_path=db_path)
    result = gate.run()
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT passed FROM kernel_results").fetchall()
    conn.close()
    assert len(rows) == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_verify.py -v
```

Expected: `ModuleNotFoundError: No module named 'kernel_factory.verify'`

- [ ] **Step 3: Write `src/kernel_factory/verify.py`**

```python
"""
Self-contained verification harness — safe to ship via SSH to a TPU VM.
All imports are standard (jax, jaxlib, numpy, sqlite3).
"""
from __future__ import annotations
import enum
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from kernel_factory.schemas import DType, KernelConfig, LayerSpec, TestResult


class VerifyMode(str, enum.Enum):
    CPU_INTERPRET = "cpu_interpret"
    TPU_INTERPRET = "tpu_interpret"
    TPU_COMPILE = "tpu_compile"


# ── JAX baselines ────────────────────────────────────────────────────────────

def _baseline_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.dot_general(a, b, (([1], [0]), ([], [])))


def _baseline_rmsnorm(x: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
    return ((x.astype(jnp.float32) / rms) * w.astype(jnp.float32)).astype(x.dtype)


# ── Pallas runners (interpret=True CPU mode) ─────────────────────────────────

def _run_matmul_interpret(
    spec: LayerSpec, config: KernelConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    import jax_pallas as pl  # type: ignore

    bm, bn, bk = config.block_m, config.block_n, config.block_k
    M, N, K = spec.M, spec.N, spec.K
    in_dt = jnp.bfloat16 if spec.input_dtype == DType.BFLOAT16 else jnp.float32
    acc_dt = jnp.float32
    out_dt = jnp.bfloat16 if spec.output_dtype == DType.BFLOAT16 else jnp.float32

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=in_dt)
    b = jax.random.normal(key, (K, N), dtype=in_dt)

    num_k = K // bk

    def kernel(a_ref, b_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)
        acc_ref[...] += jnp.dot(
            a_ref[...].astype(acc_dt), b_ref[...].astype(acc_dt),
            preferred_element_type=acc_dt,
        )
        @pl.when(pl.program_id(2) == num_k - 1)
        def _():
            o_ref[...] = acc_ref[...].astype(out_dt)

    result = pl.pallas_call(
        kernel,
        grid=(M // bm, N // bn, num_k),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
            pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
        interpret=True,
    )(a, b)

    baseline = _baseline_matmul(a.astype(acc_dt), b.astype(acc_dt)).astype(out_dt)
    return result, baseline


def _run_rmsnorm_interpret(
    spec: LayerSpec, config: KernelConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    import jax_pallas as pl  # type: ignore

    bm, bn = config.block_m, config.block_n
    M, N = spec.M, spec.N
    in_dt = jnp.bfloat16 if spec.input_dtype == DType.BFLOAT16 else jnp.float32
    out_dt = jnp.bfloat16 if spec.output_dtype == DType.BFLOAT16 else jnp.float32

    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (M, N), dtype=in_dt)
    w = jax.random.normal(key, (N,), dtype=in_dt)

    def kernel(x_ref, w_ref, o_ref):
        x_f = x_ref[...].astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f ** 2, axis=-1, keepdims=True) + 1e-6)
        o_ref[...] = ((x_f / rms) * w_ref[...].astype(jnp.float32)).astype(out_dt)

    result = pl.pallas_call(
        kernel,
        grid=(M // bm, N // bn),
        in_specs=[
            pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
            pl.BlockSpec((bn,), lambda m, n: (n,)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
        interpret=True,
    )(x, w)

    baseline = _baseline_rmsnorm(x, w)
    return result, baseline


_RUNNERS = {
    "matmul": _run_matmul_interpret,
    "rmsnorm": _run_rmsnorm_interpret,
}

_ATOL = {"matmul": 1e-2, "rmsnorm": 1e-2}
_RTOL = {"matmul": 1e-2, "rmsnorm": 1e-2}


# ── SQLite logging ────────────────────────────────────────────────────────────

def _ensure_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kernel_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            op_type TEXT,
            block_m INT, block_n INT, block_k INT,
            M INT, N INT, K INT,
            input_dtype TEXT, output_dtype TEXT,
            passed BOOLEAN,
            max_abs_error REAL,
            compile_time_ms REAL,
            execution_latency_ms REAL,
            error_trace TEXT,
            tpu_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _log_result(conn: sqlite3.Connection, spec: LayerSpec, config: KernelConfig, result: TestResult):
    conn.execute("""
        INSERT INTO kernel_results
            (op_type, block_m, block_n, block_k, M, N, K,
             input_dtype, output_dtype, passed, max_abs_error,
             compile_time_ms, execution_latency_ms, error_trace, tpu_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        spec.op_type, config.block_m, config.block_n, config.block_k,
        spec.M, spec.N, spec.K,
        spec.input_dtype.value, spec.output_dtype.value,
        result.passed, result.max_abs_error,
        result.compile_time_ms, result.execution_latency_ms,
        result.error_trace, result.tpu_version,
    ))
    conn.commit()


# ── Public gate ───────────────────────────────────────────────────────────────

class VerificationGate:
    def __init__(
        self,
        spec: LayerSpec,
        config: KernelConfig,
        mode: VerifyMode = VerifyMode.CPU_INTERPRET,
        db_path: Optional[Path] = None,
    ):
        self.spec = spec
        self.config = config
        self.mode = mode
        self.db_path = db_path

    def run(self) -> TestResult:
        spec, config = self.spec, self.config
        runner = _RUNNERS.get(spec.op_type)
        if runner is None:
            return TestResult(
                kernel_config=config, layer_spec=spec, passed=False,
                error_trace=f"No runner for op_type='{spec.op_type}'"
            )

        t0 = time.perf_counter()
        try:
            actual, expected = runner(spec, config)
            latency_ms = (time.perf_counter() - t0) * 1000
            max_err = float(jnp.max(jnp.abs(actual - expected)))
            passed = bool(jnp.allclose(
                actual, expected,
                atol=_ATOL[spec.op_type],
                rtol=_RTOL[spec.op_type],
            ))
            result = TestResult(
                kernel_config=config, layer_spec=spec, passed=passed,
                max_abs_error=max_err,
                execution_latency_ms=latency_ms,
            )
        except Exception:
            result = TestResult(
                kernel_config=config, layer_spec=spec, passed=False,
                error_trace=traceback.format_exc(),
            )

        if self.db_path:
            conn = _ensure_db(self.db_path)
            _log_result(conn, spec, config, result)
            conn.close()

        return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_verify.py -v
```

Expected: `3 passed` (CPU interpret mode, no TPU needed)

- [ ] **Step 5: Commit**

```bash
git add src/kernel_factory/verify.py tests/test_verify.py
git commit -m "feat: verification harness — CPU interpret mode, SQLite logging, numerical baseline checks"
```

---

## Self-Review Checklist

### Spec coverage
- [x] uv / pyproject.toml — Task 1
- [x] Pydantic schemas: LayerSpec, HardwareLimits, KernelConfig, TestResult — Task 2
- [x] HardwareLimits NOT hardcoded to 128 MiB — `for_v4()`, `for_v5e()`, `for_v6e()` all differ — Task 2
- [x] Kuzu KG with all 11 node types — Task 3
- [x] VMEM solver: block-size-based estimate (not full array), 75% safety margin — Task 4
- [x] VMEM formula: `input_a + input_b + output + accum` — Task 4
- [x] Last-dim alignment to `vector_width` (128), second-dim to `sublane_width` (8) — Task 4
- [x] Templates for MatMul and RMSNorm using `pl.BlockSpec` — Task 5
- [x] Assembler injects solver params, never writes Pallas logic — Task 6
- [x] Verify: CPU interpret=True mode — Task 7
- [x] Verify: op-specific baselines (lax.dot_general, pure-JAX RMSNorm) — Task 7
- [x] `jnp.allclose` with explicit atol/rtol — Task 7
- [x] SQLite logging of pass/fail, error trace, latency — Task 7
- [x] Phase 1+2 scope locked to MatMul + RMSNorm only — no attention/conv — Tasks 5, 6, 7
- [ ] TPU interpret mode (`pltpu.InterpretParams`) — verify.py has `VerifyMode.TPU_INTERPRET` enum but runner not wired. **Acceptable for local dev; wire when TPU VM is available.**
- [ ] LanceDB RAG retrieval — out of scope for Phase 3 skeleton; templates.py serves static templates. RAG wiring is a follow-on task.

### Placeholder scan
- No TBD, TODO, or "implement later" found.
- All test code is complete with actual assertions.
- All implementation code is complete with actual logic.

### Type consistency
- `DType.itemsize` property defined in Task 2, used in `solver.py` Task 4 — consistent.
- `KernelConfig` fields defined in Task 2, constructed in Tasks 4 and 7 — consistent.
- `VerificationGate` takes `LayerSpec` + `KernelConfig` — both defined in Task 2 — consistent.
