# Roadmap & Future Work

This document captures meaningful improvements that are **not yet implemented** in
kernel-factory, grouped by theme and roughly ordered by value-to-effort. Each item
points at the concrete code location it touches so it can be picked up directly.

Status legend: 🔴 not started · 🟡 partially wired · 🟢 shipped

---

## 1. Operation coverage

The pipeline currently supports `matmul`, `rmsnorm`, `fused_matmul_rmsnorm`, and
`flash_attention` (`src/kernel_factory/templates.py`, `solver.py`, `verify.py`).
The highest-leverage gap is the set of ops where XLA *cannot* cross-fuse — which is
exactly where Pallas wins (see the README benchmark note, ~0.97x on standard ops).

- 🔴 **Quantized int8 matmul.** `DType.INT8` already exists in `schemas.py` but no
  template, VMEM estimator, or verification runner uses it. Add an int8×int8→int32
  matmul with a float32 dequant epilogue. This is the clearest near-term win for
  inference workloads.
- 🔴 **MoE routing kernel.** The README explicitly names MoE as a case XLA can't
  fuse, but there is no template. Add a top-k gating + expert-dispatch kernel.
- 🔴 **Causal / masked attention + KV-cache decode.** `flash_attention` is full,
  non-causal, prefill-only. Decode (seq_len_q=1 against a cached K/V) and a causal
  mask are needed for any real autoregressive serving path.
- 🔴 **Activation & norm family:** `gelu`, `silu`, `softmax`, and standard
  `layernorm` (mean-subtracting, vs the existing RMSNorm). These are cheap templates
  and broaden the "fuse the epilogue" story (e.g. `matmul + gelu`, `matmul + silu`
  for SwiGLU FFNs).
- 🔴 **Fused SwiGLU FFN** (`gate = silu(x·Wg) * (x·Wu)` then `·Wd`) — the dominant
  FFN shape in modern LLMs (Llama, Mistral).

See `README.md` §14 "Adding a new operation" for the 5-step recipe each of these
follows (template → solver branch → VMEM formula → verify runner → tests).

## 2. Tile solver intelligence

`src/kernel_factory/solver.py` scores candidates by raw tile volume
(`block_m × block_n × block_k`) over powers of 2 plus the full dimension.

- 🔴 **Measured-latency autotuning.** Replace the static volume heuristic with an
  optional mode that compiles the top-N candidate configs and keeps the one with the
  lowest *measured* latency on the target device. Expose as `--autotune` in the CLI.
- 🔴 **Non-power-of-2 divisors.** Only powers of 2 (+ full dim) are considered, so
  N=768 can't use a 384-wide block. Generalize `_candidates()` to all hardware-aligned
  divisors of the dimension.
- 🔴 **HBM-bandwidth–aware scoring.** `HardwareLimits.hbm_bandwidth_gbps` is stored
  but never used by the solver. Use it to estimate whether a config is memory- or
  compute-bound and bias tile choice accordingly (roofline-driven).
- 🟡 **Per-generation VMEM calibration.** The VMEM estimators are hand-derived
  byte counts. Calibrate them against what the Mosaic compiler actually allocates and
  record the delta, so the 75% safety fraction can be tightened.

## 3. Verification depth

`src/kernel_factory/verify.py` runs CPU interpret mode with `allclose(atol=1e-2)`.

- 🔴 **On-TPU verification in CI.** Add an optional self-hosted/TPU job that runs the
  generated kernel on real hardware, not just interpret mode. CPU interpret does not
  exercise Mosaic tiling constraints (the source of every bug in README §15).
- 🔴 **Property-based / fuzz testing.** Sweep random valid `(M, N, K, dtype, tpu)`
  shapes through solve→assemble→verify to catch tiling edge cases automatically
  (hypothesis). The divisibility bug fixed in `f5185d7` is exactly the class this
  would have caught.
- 🔴 **Tighter, dtype-aware tolerances.** A single `atol=1e-2` is loose for fp32 and
  potentially too tight for int8. Derive tolerance from dtype + accumulation depth.
- 🔴 **Backward/gradient kernels & their verification** for training use cases.

## 4. Knowledge graph & self-repair (the unfinished half)

The Kuzu schema (`src/kernel_factory/kg/schema.py`) defines a rich provenance graph
including `FailureCase → KnownBug → FixPattern` edges — but it is never connected.
`cli.py` and `pipeline.py` pass `kg=None`, so nothing is persisted or queried.

- 🟡 **Wire the KG into the CLI/pipeline.** Add a `--kg-path` option that persists
  every generation (spec, config, code, test result) into the graph.
- 🔴 **Closed-loop auto-repair.** On a verification failure, query the graph for a
  matching `KnownBug`/`FixPattern` (e.g. the bf16-output-<512 and 1D-weight retiling
  bugs in README §15 are already documented patterns) and re-assemble with the fix
  applied — then re-verify. This turns the static "known constraints" table into an
  automated recovery path.

## 5. Self-improving RAG corpus

`ProductionRAG` (`src/kernel_factory/rag.py`) falls back to static templates when the
corpus is empty and the shipped corpus is small.

- 🔴 **Feed verified kernels back into the corpus.** Every kernel that passes the
  verification gate is a high-quality retrieval example — ingest it automatically so
  retrieval quality compounds over time.
- 🔴 **Grow the corpus** from public Pallas/Mosaic kernels via `scripts/ingest_rag.py`,
  with `kernel_class` metadata for all op families above.

## 6. Benchmarking & hardware breadth

- 🔴 **Beyond GPT-2 small.** `scripts/benchmark_gpt2.py` covers one model. Add Llama-2
  7B and Mistral 7B layer shapes, plus a roofline / % -of-peak-FLOPs report.
- 🔴 **Regression tracking.** Persist benchmark numbers per commit and fail CI if a
  kernel regresses beyond a threshold.
- 🔴 **More targets.** Add `v5p`; correct the per-generation VMEM (v5e/v6e are
  hardcoded to 16 MiB in `schemas.py`). Longer term: a Triton/GPU backend behind the
  same `LayerSpec → KernelConfig` interface.
- 🔴 **Multi-chip / sharding** awareness for layers that don't fit a single chip.

## 7. Project hygiene & packaging

- 🔴 **LICENSE file.** `README.md` says "See repository for license terms" but no
  `LICENSE` exists — blocks any external adoption. Pick and add one.
- 🔴 **CONTRIBUTING.md** documenting the add-an-op recipe and the test workflow.
- 🔴 **Lint/format/type gates** (ruff + mypy) wired into the new CI workflow and a
  pre-commit config.
- 🔴 **Release automation** — build + publish to PyPI on tag; the package is already
  structured for it (`pyproject.toml` defines the `kernel-factory` entry point).

## 8. Batch / model-level UX

- 🔴 **Generate-a-whole-model command.** Take a HuggingFace config (hidden size, heads,
  layers, FFN dim) and emit the full set of verified kernels for that architecture in
  one call, instead of one shape at a time.
- 🔴 **`--target-latency` mode** that autotunes (see §2) until a latency budget is met
  or reports the best achievable.

---

*Shipped recently:* solver now guarantees tiles evenly divide their dimensions
(commit `f5185d7`); Python test suite runs in CI (`.github/workflows/ci.yml`).
