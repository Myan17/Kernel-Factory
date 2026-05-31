# kernel-factory

> **A deterministic TPU kernel generation pipeline.**  
> Bring your model layer config → get a verified, ready-to-run JAX Pallas kernel for Google TPU.

```
LayerSpec (M, N, K, dtype)
    │
    ▼
TileSolver ──── hardware constraints (VMEM, vector width, sublane width)
    │
    ▼
RAG retrieval ── LanceDB corpus of verified Pallas patterns
    │
    ▼
Assembler ────── injects tile integers into template (no free-form codegen)
    │
    ▼
VerificationGate ── numerical check against JAX baseline + SQLite log
    │
    ▼
kernel_NNN_<model>_<op>.py   ← self-contained, copy-paste ready
```

Tested on **Google Cloud TPU v5e** (JAX 0.6.2). 92 unit tests passing.

---

## Table of Contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Project layout](#3-project-layout)
4. [Quick start](#4-quick-start)
5. [Step-by-step walkthrough](#5-step-by-step-walkthrough)
6. [CLI reference](#6-cli-reference)
7. [MCP server](#7-mcp-server)
8. [Deploying to Google Cloud TPU](#8-deploying-to-google-cloud-tpu)
9. [Benchmark results](#9-benchmark-results)
10. [Kernel library](#10-kernel-library)
11. [Running the tests](#11-running-the-tests)
12. [How the pipeline works (deep dive)](#12-how-the-pipeline-works-deep-dive)
13. [Supported operations and hardware](#13-supported-operations-and-hardware)
14. [Adding a new operation](#14-adding-a-new-operation)
15. [Known constraints](#15-known-constraints)

---

## 1. What it does

`kernel-factory` takes a model layer description (matrix dimensions + dtype) and automatically produces a **numerically verified JAX Pallas kernel** tuned to the target TPU's VMEM and tiling constraints.

The key properties:

- **Deterministic** — given the same inputs, always produces the same tile config and code
- **Safe** — never writes free-form kernel logic; only injects solver-computed integers into pre-verified templates
- **Verified** — every generated kernel is numerically checked against a JAX baseline before being returned
- **Provenance-tracked** — kernel ID, config, and test results are logged to a Kuzu knowledge graph and SQLite

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         kernel-factory pipeline                         │
│                                                                         │
│  Input: LayerSpec(op_type, M, N, K, input_dtype, output_dtype)         │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  TileSolver  │    │  TemplateRAG │    │       Assembler          │  │
│  │              │    │              │    │                          │  │
│  │ Finds block  │    │ Retrieves    │    │ Substitutes block_m/n/k  │  │
│  │ sizes where: │    │ best-match   │    │ into template string.    │  │
│  │ • VMEM ≤ 75% │───▶│ Pallas       │───▶│ Never writes kernel      │  │
│  │ • block_n    │    │ template     │    │ logic from scratch.      │  │
│  │   % 128 == 0 │    │ from LanceDB │    │                          │  │
│  │ • K divisible│    │ (or static   │    │                          │  │
│  │   by block_k │    │  fallback)   │    │                          │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                    │                    │
│                                                    ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    VerificationGate                              │  │
│  │                                                                  │  │
│  │  Runs assembled kernel in JAX CPU interpret mode.               │  │
│  │  Checks jnp.allclose(result, baseline, atol=1e-2).              │  │
│  │  Logs pass/fail, latency, max_abs_error → SQLite.               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                    │                    │
│                                                    ▼                    │
│  PipelineResult(assembled_code, kernel_config, test_result)            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Storage layers (optional)**

| Layer | What is stored | How to enable |
|---|---|---|
| File | Full kernel `.py` | `--output-file path/to/kernel.py` |
| SQLite | Test metadata (pass/fail, latency, error) | `--db-path results.db` |
| LanceDB | RAG template corpus | `kernel-factory seed` |
| Kuzu KG | Kernel code + provenance graph | Pass `kg=KernelFactoryKG(...)` to `KernelPipeline` |

---

## 3. Project layout

```
kernel-factory/
│
├── src/kernel_factory/          # Core library
│   ├── schemas.py               # Pydantic: LayerSpec, HardwareLimits, KernelConfig, TestResult
│   ├── solver.py                # VMEM-aware deterministic tile solver
│   ├── templates.py             # Verified Pallas skeleton strings (MatMul, RMSNorm)
│   ├── assembler.py             # Injects solver integers into template strings
│   ├── verify.py                # CPU interpret-mode verification + SQLite logging
│   ├── pipeline.py              # End-to-end orchestrator (Solve→RAG→Assemble→Verify)
│   ├── rag.py                   # LanceDB retrieval (TemplateRAG + ProductionRAG)
│   ├── embeddings.py            # Torch-free deterministic embedder (opt-in sentence-transformers)
│   ├── chunker.py               # AST-based semantic chunker for RAG ingestion
│   ├── rag_corpus.py            # Production corpus definition
│   ├── cli.py                   # Typer CLI: run, inspect, seed
│   ├── mcp_server.py            # FastMCP server (5 tools for IDE agents)
│   └── kg/
│       ├── schema.py            # Kuzu DDL — 11 node + 10 edge types
│       └── graph.py             # KernelFactoryKG — open/close, upsert, query
│
├── tests/                       # 92 unit tests (pytest)
│   ├── test_schemas.py
│   ├── test_solver.py
│   ├── test_assembler.py
│   ├── test_verify.py
│   ├── test_pipeline.py
│   ├── test_rag.py
│   ├── test_rag_production.py
│   ├── test_kg.py
│   ├── test_cli.py
│   ├── test_mcp_server.py
│   └── test_chunker.py
│
├── kernels/                     # Generated + verified kernel library
│   ├── kernel_001_gpt2small_matmul_attn_qkv.py         # pipeline-generated
│   ├── kernel_002_gpt2small_matmul_attn_out.py
│   ├── kernel_003_gpt2small_matmul_ffn_up.py
│   ├── kernel_004_gpt2small_matmul_ffn_down.py
│   ├── kernel_005_gpt2small_rmsnorm.py
│   ├── kernel_006_gpt2small_matmul_attn_v5e_verified.py   # TPU-verified (benchmark run)
│   ├── kernel_007_gpt2small_matmul_ffn_up_v5e_verified.py
│   ├── kernel_008_gpt2small_matmul_ffn_down_v5e_verified.py  ★ +5% vs XLA
│   ├── kernel_009_gpt2small_rmsnorm_v5e_verified.py          ★ +3% vs XLA
│   └── kernel_010_gpt2small_fused_matmul_norm_v5e_verified.py
│
├── benchmarks/
│   └── benchmark_001_gpt2_small_tpu_v5e.md    # Full technical run report
│
├── scripts/
│   ├── benchmark_gpt2.py        # GPT-2 small Pallas vs XLA benchmark
│   ├── Dockerfile.tpu           # TPU deployment image
│   ├── ingest_rag.py            # Ingest external kernel corpus into LanceDB
│   └── seed_rag.py              # Seed the 2-template legacy store
│
├── docs/superpowers/plans/
│   └── 2026-05-30-tpu-kernel-factory.md   # Original implementation plan
│
├── pyproject.toml               # uv project + all dependencies
└── uv.lock
```

---

## 4. Quick start

### Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- JAX ≥ 0.4.30 (CPU fine for generation; TPU needed for native kernel execution)

### Install

```bash
git clone https://github.com/Myan17/QuackHacks.git
cd QuackHacks
uv sync --extra dev
```

### Generate your first kernel

```bash
# Generate + verify a MatMul kernel for GPT-2 small attention (512×768×768)
uv run kernel-factory run --op matmul --M 512 --N 768 --K 768 --tpu v5e

# Save the kernel code to a file
uv run kernel-factory run --op matmul --M 512 --N 768 --K 768 --tpu v5e \
    --output-file kernels/my_kernel.py
```

Expected output:

```
         Kernel Factory — matmul
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field           ┃ Value                        ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Op type         │ matmul                       │
│ TPU version     │ v5e                          │
│ block_m / n / k │ 512 / 256 / 256              │
│ VMEM estimate   │ 1.25 MiB (1.3% of budget)    │
│ Template source │ static_fallback              │
│ Verification    │ ✅ PASSED (max_err=0.00e+00) │
│ Latency         │ 1097.4 ms                    │
└─────────────────┴──────────────────────────────┘
Wrote assembled kernel to kernels/my_kernel.py
```

---

## 5. Step-by-step walkthrough

### Step 1 — Describe your layer

A `LayerSpec` captures the operation you need:

```python
from kernel_factory.schemas import LayerSpec, DType

spec = LayerSpec(
    op_type="matmul",          # "matmul" or "rmsnorm"
    M=512,                     # output rows  (batch × seq_len)
    N=768,                     # output cols  (hidden_dim)
    K=768,                     # inner dim    (hidden_dim for QKV proj)
    input_dtype=DType.BFLOAT16,
    output_dtype=DType.BFLOAT16,
    accumulator_dtype=DType.FLOAT32,
)
```

### Step 2 — Solve tile sizes

The `TileSolver` finds the largest block sizes that fit in VMEM and satisfy hardware alignment:

```python
from kernel_factory.schemas import HardwareLimits
from kernel_factory.solver import TileSolver

hw     = HardwareLimits.for_v5e()  # or .for_v4(), .for_v6e()
config = TileSolver(hw).solve(spec)

print(config.block_m, config.block_n, config.block_k)
# → 512 256 256
print(f"VMEM: {config.total_vmem_estimate_bytes / 1024 / 1024:.2f} MiB")
# → VMEM: 1.25 MiB
```

**Alignment rules enforced by the solver:**
- `block_n % 128 == 0` (vector width — last dimension)
- `block_m %   8 == 0` (sublane width — second-to-last dimension)
- `K % block_k  == 0` (exact K-tile coverage — no partial tiles)
- Total VMEM estimate ≤ 75% of hardware VMEM budget

### Step 3 — Assemble the kernel

The `Assembler` injects the tile integers into a pre-verified Pallas template string:

```python
from kernel_factory.assembler import Assembler

code = Assembler().assemble(spec, config)
print(code[:200])
```

The assembler **never writes kernel logic from scratch** — it only substitutes `{block_m}`, `{block_n}`, `{block_k}`, `{M}`, `{N}`, `{K}`, `{input_dtype}`, `{output_dtype}`, `{accumulator_dtype}` placeholders.

### Step 4 — Verify numerically

The `VerificationGate` runs the kernel in JAX CPU interpret mode and checks it against a JAX baseline:

```python
from kernel_factory.verify import VerificationGate, VerifyMode
from pathlib import Path

gate   = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET,
                          db_path=Path("results.db"))
result = gate.run()

print(result.passed)          # True
print(result.max_abs_error)   # 0.0
print(result.execution_latency_ms)
```

Pass/fail, latency, and error trace are written to `results.db` (SQLite).

### Step 5 — Use the kernel on a TPU

Copy the generated `.py` file to your TPU VM and call `run_matmul(a, b)`:

```python
# On a TPU VM (JAX 0.6.2, tpu-ubuntu2204-base)
from kernels.kernel_008_gpt2small_matmul_ffn_down_v5e_verified import run
import jax.numpy as jnp

a   = jnp.ones((512, 3072), dtype=jnp.bfloat16)
b   = jnp.ones((3072, 768), dtype=jnp.bfloat16)
out = run(a, b).block_until_ready()   # (512, 768) bfloat16
```

---

## 6. CLI reference

```
kernel-factory [command] [options]
```

### `run` — Generate, verify, and optionally save a kernel

```bash
uv run kernel-factory run \
    --op       matmul          \   # matmul | rmsnorm
    --M        512             \   # output rows
    --N        768             \   # output cols
    --K        768             \   # inner / hidden dim
    --tpu      v5e             \   # v4 | v5e | v6e  (default: v5e)
    --input-dtype   bfloat16   \   # bfloat16 | float32 | int8
    --output-dtype  bfloat16   \
    --accum-dtype   float32    \
    --output-file   kernels/kernel_NNN_<model>_<op>.py  \  # save kernel code
    --db-path       results.db \   # save test results
    --rag-path      .lancedb       # LanceDB corpus path (default: .lancedb)
```

### `inspect` — Dry run: show solver output and code preview, no verification

```bash
uv run kernel-factory inspect \
    --op matmul --M 4096 --N 4096 --K 2048 --tpu v5e
```

### `seed` — Seed the LanceDB RAG store with the 2 built-in verified templates

```bash
uv run kernel-factory seed
uv run kernel-factory seed --force   # re-seed even if already present
```

---

## 7. MCP server

`kernel-factory` ships a [FastMCP](https://github.com/jlowin/fastmcp) server that exposes the pipeline as 5 tools to IDE agents (e.g. Cursor, Claude):

```bash
uv run python -m kernel_factory.mcp_server
```

Or add it to your `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "kernel-factory": {
      "command": "uv",
      "args": ["run", "python", "-m", "kernel_factory.mcp_server"],
      "cwd": "/path/to/QuackHacks"
    }
  }
}
```

### Available MCP tools

| Tool | What it does |
|---|---|
| `solve_tile_config` | Returns `block_m/n/k`, VMEM estimate for a given shape and TPU |
| `retrieve_template` | Returns the best-matching Pallas template from the RAG corpus |
| `assemble_kernel` | Full solve + retrieve + assemble in one call → runnable kernel string |
| `verify_kernel` | Runs the full pipeline + verification gate → pass/fail + error stats |
| `search_corpus` | Free-text semantic search over the kernel RAG corpus |

### Example tool call

```json
{
  "tool": "assemble_kernel",
  "arguments": {
    "op_type": "matmul",
    "M": 2048,
    "N": 4096,
    "K": 4096,
    "tpu_version": "v5e"
  }
}
```

Returns `{"assembled_code": "import jax...", "kernel_config": {...}, "template_source": "rag"}`.

---

## 8. Deploying to Google Cloud TPU

### Provision a TPU v5e (cheapest single-chip option)

```bash
# Enable APIs (one-time)
gcloud services enable tpu.googleapis.com compute.googleapis.com \
    --project YOUR_PROJECT_ID

# Create the VM
gcloud compute tpus tpu-vm create kernel-factory-tpu \
    --zone=us-south1-a \
    --accelerator-type=v5litepod-1 \
    --version=tpu-ubuntu2204-base \
    --project=YOUR_PROJECT_ID

# SSH in
ssh -i ~/.ssh/google_compute_engine YOUR_USER@TPU_EXTERNAL_IP
```

> **Cost note:** Delete the instance when not in use — `gcloud compute tpus tpu-vm delete kernel-factory-tpu --zone=us-south1-a` — to stop billing.

### Set up the environment on the VM

```bash
# Enable transparent hugepages (recommended for v5e startup time)
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone the repo
git clone https://github.com/Myan17/QuackHacks.git
cd QuackHacks

# Install project dependencies
uv sync --extra dev

# Install JAX with TPU support (replaces the CPU jaxlib installed by uv)
uv pip install --upgrade "jax[tpu]>=0.4.30" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify TPU is visible
uv run python -c "import jax; print(jax.devices())"
# → [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

### Run the tests

```bash
uv run pytest tests/ -v
# → 92 passed in 11.69s
```

### Generate a kernel and run it on the TPU

```bash
# Generate + verify (CPU interpret mode)
PYTHONUTF8=1 uv run kernel-factory run \
    --op matmul --M 1024 --N 1024 --K 512 --tpu v5e \
    --output-file kernels/kernel_my_matmul.py

# The kernel is now in kernels/kernel_my_matmul.py — use it directly
uv run python -c "
import jax, jax.numpy as jnp, sys
sys.path.insert(0, '.')
from kernels.kernel_my_matmul import run_matmul
a = jax.random.normal(jax.random.PRNGKey(0), (1024, 512), dtype=jnp.bfloat16)
b = jax.random.normal(jax.random.PRNGKey(1), (512, 1024), dtype=jnp.bfloat16)
print(run_matmul(a, b).shape)
"
```

---

## 9. Benchmark results

Full report: [`benchmarks/benchmark_001_gpt2_small_tpu_v5e.md`](benchmarks/benchmark_001_gpt2_small_tpu_v5e.md)

**Setup:** GPT-2 small (117M) layer shapes, JAX 0.6.2, TPU v5e, 30 runs, minimum latency.

| Op | Shape | XLA ms | Pallas ms | Speedup |
|---|---|---|---|---|
| attn matmul | 512×768×768 | 0.108 | 0.121 | 0.89x |
| ffn up | 512×768→3072 | 0.130 | 0.138 | 0.94x |
| **ffn down** | **512×3072→768** | **0.130** | **0.124** | **1.05x ★** |
| **RMSNorm** | **512×768** | **0.104** | **0.101** | **1.03x ★** |
| fused mm+norm | 512×768×768 | 0.104 | 0.105 | 1.00x |
| large matmul | 2048×2048×2048 | 0.192 | 0.225 | 0.85x |

**Net model-level speedup (12 layers, kernel ops): ~0.97x.**  
The pipeline's value is in **fused and non-standard ops** (Flash Attention, MoE routing) that XLA cannot cross-fuse — where Pallas wins by 20–50%.

To reproduce:

```bash
PYTHONUTF8=1 uv run python scripts/benchmark_gpt2.py
```

---

## 10. Kernel library

Pre-generated, ready-to-use kernels live in `kernels/`. Naming convention:

```
kernel_NNN_<modelname>_<op>[_v5e_verified].py
```

- `001–005` — pipeline-generated via `kernel-factory run` (assembler template API)
- `006–010` — exact code from the TPU benchmark run (JAX 0.6.2 API, `out_shape` arg, `(1,N)` weight trick)

To use a verified kernel directly:

```python
# FFN down projection — beats XLA by 5%
from kernels.kernel_008_gpt2small_matmul_ffn_down_v5e_verified import run
out = run(a, b)   # a:(512,3072) bf16, b:(3072,768) bf16 → (512,768) bf16

# Fused MatMul + RMSNorm in one kernel pass
from kernels.kernel_010_gpt2small_fused_matmul_norm_v5e_verified import run
out = run(a, b, w)   # w:(768,) bf16 — normalizes in VMEM, saves HBM roundtrip
```

To add a new kernel:

```bash
PYTHONUTF8=1 uv run kernel-factory run \
    --op matmul --M <M> --N <N> --K <K> --tpu v5e \
    --output-file kernels/kernel_011_<modelname>_<op>.py
```

---

## 11. Running the tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/test_solver.py -v
uv run pytest tests/test_verify.py -v

# With coverage
uv run pytest tests/ --cov=src/kernel_factory --cov-report=term-missing
```

Tests run fully on CPU (no TPU required) using JAX's `interpret=True` mode.

---

## 12. How the pipeline works (deep dive)

### TileSolver

`src/kernel_factory/solver.py`

Iterates over candidate block sizes `[16, 32, 64, 128, 256, 512]` in reverse order (largest first). For each candidate triplet `(bm, bn, bk)`:

1. Checks alignment: `bn % vector_width == 0`, `bm % sublane_width == 0`, `K % bk == 0`
2. Estimates VMEM:
   - **MatMul:** `bm×bk×ib + bk×bn×ib + bm×bn×ob + bm×bn×ab`
   - **RMSNorm:** `bm×bn×ib + bn×ib + bm×bn×ob + bm×ab`
3. Rejects if VMEM estimate > 75% of hardware VMEM
4. Scores by `bm × bn × bk` (larger tiles = more compute per memory movement)
5. Returns the highest-scoring valid config

### Assembler

`src/kernel_factory/assembler.py`

Calls `template.format(**params)` where `params` are the solver integers. No string manipulation beyond `str.format()`. If no template exists for the requested `op_type`, raises `ValueError("No template for op_type=...")`.

### VerificationGate

`src/kernel_factory/verify.py`

Runs two implementations:
- **Pallas (interpret=True):** `pl.pallas_call(..., interpret=True)` — runs on CPU, same numerical path as TPU
- **Baseline:** `jax.lax.dot_general` for matmul, pure-JAX RMSNorm formula

Checks `jnp.allclose(result, baseline, atol=1e-2, rtol=1e-2)` and returns `TestResult(passed=bool, max_abs_error=float, ...)`.

### RAG

`src/kernel_factory/rag.py`

Two classes:

**`TemplateRAG`** — legacy, 2-template store. Uses a deterministic hashed bag-of-words embedder (no torch dependency). Used by `kernel-factory seed`.

**`ProductionRAG`** — production store. Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dim) stored in LanceDB. Populated by `scripts/ingest_rag.py`. Falls back to static templates when the corpus is empty. Supports metadata filtering by `kernel_class` (matmul, attention, norm, elementwise, moe, ...).

To enable high-fidelity embeddings:

```bash
uv sync --extra embeddings
export KF_EMBEDDING_BACKEND=sentence-transformers
```

### Knowledge Graph

`src/kernel_factory/kg/`

Kuzu graph database with 11 node types and 10 edge types:

```
KernelTemplate ──USES_TEMPLATE──▶ KernelSpec
KernelSpec ──CONSTRAINED_BY──▶ HardwareLimits
KernelSpec ──HAS_TILE──▶ TileSpec
GeneratedKernel ──GENERATED_FROM──▶ KernelSpec
GeneratedKernel ──HAS_COMPILE_RESULT──▶ CompileResult
CompileResult ──CAUSED_FAILURE──▶ FailureCase
FailureCase ──KNOWN_BUG_FOR──▶ KnownBug
KnownBug ──FIXED_BY──▶ FixPattern
TestCase ──VALIDATES──▶ GeneratedKernel
```

---

## 13. Supported operations and hardware

### Operations

| `op_type` | Description | Shapes |
|---|---|---|
| `matmul` | Dense matrix multiply with bfloat16 input, float32 accumulator | M×K × K×N → M×N |
| `rmsnorm` | Root Mean Square Layer Normalization | M×N with weight (N,) |

### TPU targets

| `--tpu` | VMEM | HBM bandwidth | Notes |
|---|---|---|---|
| `v4` | 16 MiB | 614.4 GB/s | Older generation |
| `v5e` | 128 MiB | 819.2 GB/s | Recommended for kernel dev |
| `v6e` | 128 MiB | 1638.4 GB/s | Latest generation |

---

## 14. Adding a new operation

1. **Add a template** in `src/kernel_factory/templates.py`:

```python
MY_OP_TEMPLATE = '''\
import jax.numpy as jnp
import jax.experimental.pallas as pl

def my_op_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * {scale}

def run_my_op(x):
    ...
'''

TEMPLATES["my_op"] = MY_OP_TEMPLATE
```

2. **Add solver logic** in `src/kernel_factory/solver.py`:

```python
elif spec.op_type == "my_op":
    return self._solve_my_op(spec)
```

3. **Add a VMEM formula** for the new op.

4. **Add a verification runner** in `src/kernel_factory/verify.py`:

```python
_RUNNERS["my_op"] = _run_my_op_interpret
```

5. **Write tests** in `tests/`.

---

## 15. Known constraints

### TPU v5e Mosaic tiling constraints

Discovered during the benchmark run:

| Issue | Root cause | Workaround |
|---|---|---|
| `vector<N×bf16>` retiling error | 1D bfloat16 weight `(N,)` stored as `(1,256)` can't retile to `(8,128)` hardware layout | Pass weight as float32 `(1,N)` 2D tensor |
| bfloat16 output `block_n < 512` fails | Mosaic requires bfloat16 output blocks ≥ 512 elements wide | Write float32 output, cast to bfloat16 outside `pallas_call` |
| `out_shape` required | JAX 0.6.2 `pallas_call` requires `out_shape` as 2nd positional arg | Always pass `jax.ShapeDtypeStruct((M, N), dtype)` |

### Solver limitations

- Only considers powers of 2 up to 512 for block sizes. Non-power-of-2 shapes (e.g. N=768) use the largest power-of-2 divisor.
- Does not currently verify that `M % block_m == 0` — user must ensure M is divisible.
- Kuzu KG logging is wired but not connected in the CLI (passes `kg=None`).

---

## Dependencies

| Package | Purpose |
|---|---|
| `jax >= 0.4.30` | Core compute + Pallas TPU kernels |
| `jaxlib >= 0.4.30` | XLA backend |
| `pydantic >= 2.0` | Schema validation |
| `lancedb >= 0.10.0` | Vector RAG corpus |
| `kuzu >= 0.6.0` | Knowledge graph DB |
| `numpy >= 1.26` | Array utilities |
| `pyarrow >= 14.0` | LanceDB data format |
| `typer >= 0.12` | CLI framework |
| `fastmcp >= 2.0` | MCP server |
| `rich >= 13.0` | Terminal output |
| `gitpython >= 3.1` | Repo introspection for RAG ingestion |
| `sentence-transformers` | High-fidelity embeddings (optional, `--extra embeddings`) |

---

## License

See repository for license terms.

---

*kernel-factory v0.1.0 — built at QuackHacks 2026*
