# Contributing to kernel-factory

Thanks for your interest! This guide covers the dev setup, the test workflow, and
the recipe for the most common contribution — adding a new operation.

## Dev setup

```bash
git clone <repo-url>
cd kernel-factory
uv sync --extra dev        # core + pytest
# optional, for high-fidelity RAG embeddings:
uv sync --extra embeddings
```

No TPU is required for development — the test suite and the verification gate run
entirely on CPU via JAX `interpret=True` mode.

## Running the tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_solver.py -v                       # one module
uv run pytest tests/ --cov=src/kernel_factory --cov-report=term-missing
```

CI runs this same suite on Python 3.10 and 3.12 for every push and PR
(`.github/workflows/ci.yml`). Please make sure tests pass locally before opening a PR.

## Adding a new operation

Each op flows through the same five extension points. Use an existing op
(`matmul`, `rmsnorm`, `fused_matmul_rmsnorm`, `flash_attention`) as a worked example.

1. **Template** — add a verified Pallas skeleton string in
   `src/kernel_factory/templates.py` and register it in the `TEMPLATES` dict.
   Use only `{block_m}`, `{block_n}`, `{block_k}`, `{M}`, `{N}`, `{K}`, and the dtype
   placeholders. The assembler never writes kernel logic from scratch — it only
   substitutes these integers.
2. **Solver branch** — add a `_solve_<op>()` method in `src/kernel_factory/solver.py`
   and dispatch to it from `TileSolver.solve()`.
3. **VMEM estimator** — add a `_vmem_<op>()` function next to the others in
   `solver.py`. Count every buffer that lives in VMEM, double-buffered where the
   pipeline overlaps it.
4. **Verification runner** — add a CPU interpret-mode runner and a pure-JAX baseline
   in `src/kernel_factory/verify.py`, then register it in `_CPU_RUNNERS`.
5. **Tests** — add coverage under `tests/`. At minimum: the solver returns a config
   that fits the VMEM budget, every tile evenly divides its dimension, and the
   verification gate passes against the baseline.

Don't forget to expose the new `op_type` in the CLI's `_SUPPORTED_OPS`
(`src/kernel_factory/cli.py`) and, if relevant, the MCP server.

## Commit & PR conventions

- Conventional-commit style prefixes (`feat:`, `fix:`, `docs:`, `ci:`, `refactor:`).
- One logical change per commit; keep generated kernels and source changes separate.
- Open a PR against `main`; CI must be green before merge.

## Where to start

See [`ROADMAP.md`](ROADMAP.md) for prioritized, code-anchored ideas — int8 matmul,
MoE routing, causal/decode attention, and solver autotuning are all good first
contributions.
