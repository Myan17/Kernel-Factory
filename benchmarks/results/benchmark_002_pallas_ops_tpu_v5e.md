# Benchmark 002 — Pallas Ops on TPU v5e (v5litepod-1)

**Date:** 2026-05-31  
**Hardware:** Google Cloud TPU v5litepod-1 (single chip, 1 TensorCore, 16 MiB VMEM)  
**JAX version:** 0.6.2  
**Device:** `TPU_0(process=0,(0,0,0,0))`  
**Warmup:** 5 iterations · **Measurement:** 50 iterations (median)

---

## Summary

| Section | Result | Highlight |
|---------|--------|-----------|
| A — Standalone MatMul | ✅ Parity | All shapes within 0.96–1.00× of XLA |
| B — Fused MatMul+RMSNorm | ✅ Win (large batch) | Up to **1.11×** at batch=8192 |
| C — FlashAttention | ⚠️ Mixed | 1.03–1.05× short seq; slower for seq≥2048 |

---

## Section A — Standalone MatMul

**Target:** speedup ≥ 0.98× (parity with XLA)

| Shape | XLA ms | Pallas ms | Speedup |
|-------|--------|-----------|---------|
| gpt2 attn_qkv (512×768×768) | 0.122 | 0.121 | **1.00×** ✅ |
| gpt2 ffn_up (512×768→3072) | 0.131 | 0.131 | **1.00×** ✅ |
| med attn (2048×768×768) | 0.130 | 0.135 | 0.96× ⚠️ |
| med ffn_up (2048×768→3072) | 0.173 | 0.173 | **1.00×** ✅ |
| large matmul (1024×1024×1024) | 0.132 | 0.132 | **1.00×** ✅ |

**Notes:**  
The pipeline-generated Pallas kernels achieve near-perfect parity with XLA's optimised matmul across all GPT-2 representative shapes. The `2048×768×768` shape is 0.96× — a small regression attributable to the solver's tile selection producing suboptimal double-buffering for that specific aspect ratio.

---

## Section B — Fused MatMul+RMSNorm

**Approach:** Pallas fuses the matmul K-reduction with RMSNorm into a single kernel. The float32 intermediate result (M×N) never touches HBM — it is normalised directly in VMEM.

**Target:** speedup ≥ 1.05× at batch ≥ 4096

| Shape | XLA 2-op ms | Pallas fused ms | Speedup | HBM saved |
|-------|------------|-----------------|---------|-----------|
| batch=512 (512×1024×1024) | 0.126 | 0.127 | 0.99× | 1.0 MB |
| batch=2048 (2048×1024×1024) | 0.151 | 0.146 | **1.03×** | 4.0 MB |
| batch=4096 (4096×1024×1024) | 0.184 | 0.172 | **1.07×** ✅ | 8.0 MB |
| batch=8192 (8192×1024×1024) | 0.245 | 0.220 | **1.11×** ✅ | 16.0 MB |

**Notes:**  
The fusion benefit scales predictably with batch size. At batch=512, the intermediate tensor (1 MB) fits comfortably in HBM cache so XLA is competitive. At batch=8192, the 16 MB intermediate exceeds the L2 cache and the HBM round-trip becomes the bottleneck — the fused Pallas kernel eliminates it entirely, achieving 1.11× speedup.

---

## Section C — FlashAttention

**Approach:** Pallas keeps the Q×K score matrix in VMEM across the k-loop. The softmax numerics (running max `m` and normaliser `l`) are maintained as scratch buffers.

**Target:** speedup ≥ 1.3× at seq ≥ 1024

| Shape | XLA ms | Flash ms | Speedup | Score matrix |
|-------|--------|----------|---------|--------------|
| seq=512 h=12 d=64 | 0.154 | 0.149 | **1.03×** | 12.0 MB |
| seq=1024 h=12 d=64 | 0.189 | 0.181 | **1.05×** | 48.0 MB |
| seq=2048 h=12 d=64 | 0.261 | 0.441 | 0.59× ❌ | 192.0 MB |
| seq=4096 h=12 d=64 | 0.401 | 1.237 | 0.32× ❌ | 768.0 MB |

**Notes:**  
At short sequences (512, 1024), the Pallas kernel is marginally faster than XLA's `jax.nn.dot_product_attention`. At longer sequences (2048+), the kernel is significantly slower. The cause is a tile size issue: with `head_dim=64`, the solver selects small `block_k` tiles (≈64), creating O(seq/block_k) kernel iterations per Q-block. At seq=4096 this is 64 iterations, each with expensive exponential operations and no pipelining benefit. JAX's XLA-fused attention path handles this via more aggressive horizontal fusion.

**Improvement path:** Increase `block_k` up to `seq_len` for short `head_dim` cases, or implement a causal mask that allows aggressive k-loop fusion.

---

## Pipeline Health

All kernels were generated end-to-end by the `kernel-factory` pipeline:
1. **TileSolver** — deterministic tile size selection using VMEM budget (12 MiB, 75% of 16 MiB hardware limit)
2. **TemplateRAG** — op-type-filtered template retrieval from LanceDB
3. **Assembler** — template substitution (tile sizes, dtypes, scale factors)
4. **VerificationGate** — CPU interpret correctness check before execution

No kernel was written by hand. All 13 benchmarked shapes used pipeline-generated code.

---

## Key Engineering Findings

| Finding | Resolution |
|---------|-----------|
| Mosaic scoped VMEM = 16 MiB (not 128 MiB as initially assumed) | Fixed `HardwareLimits.for_v5e()` |
| Mosaic double-buffers the accumulator when `stages=2` | Updated `_vmem_matmul` estimator to use `2×acc` |
| `("parallel","parallel","arbitrary")` materialises full M×N in VMEM | Changed to `("parallel","arbitrary","arbitrary")` |
| 1D weight BlockSpec `(N,)` causes Mosaic tiling change error | Broadcast to 2D `(block_m, N)` before Pallas call |
| `jnp.dot` on 4D tensors uses wrong contraction dims | Use 2D slicing `q_ref[0,0,:,:]` inside kernel |
| Float32 accumulation cannot be written to bf16 `o_ref` | Added dedicated float32 `o_acc_ref` scratch buffer |
