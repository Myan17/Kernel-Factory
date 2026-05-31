"""
kernel_010 — GPT-2 small Fused MatMul + RMSNorm  (single Pallas pass)
Verified on: Google Cloud TPU v5e (v5litepod-1, us-south1-a)
JAX version: 0.6.2
Benchmark:   XLA (2 ops) 0.104 ms  →  Pallas fused 0.105 ms (1.00x)

Shape: M=512, N=768, K=768  (seq_len × hidden → hidden, then normalize)
Tiles: block_m=512, N=768 (full), block_k=256  (VMEM ~2.2 MiB / 2.3% of budget)

How it works:
  - Grid: (M//BM, K//BK) — no N dimension; full N columns processed per tile
  - Accumulates A×B across K tiles into VMEM scratch (stays off HBM entirely)
  - On the final K tile: applies RMSNorm in-place before writing the result
  - Saves 1 full HBM write of (M,N) + 1 full HBM read that separate ops need
  - Output is float32; caller casts to bfloat16

Key v5e Mosaic notes (same as kernel_009):
  - Weight w passed as float32 (1, N) to avoid bfloat16 1D retiling error
  - Output written as float32 to avoid N=768 bfloat16 block constraint
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M,  N,  K  = 512, 768, 768
BM, BK     = 512, 256
NUM_K      = K // BK   # 3
EPS        = 1e-6


def _kernel(a_ref, b_ref, w_ref, o_ref, acc_ref):
    # ── Step 1: accumulate K tiles of A×B into VMEM ──────────────────────────
    @pl.when(pl.program_id(1) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)   # (BM, N) float32 in VMEM

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )

    # ── Step 2: on last K tile, fuse RMSNorm before writing to HBM ───────────
    @pl.when(pl.program_id(1) == NUM_K - 1)
    def _():
        x   = acc_ref[...]                              # (BM, N) float32
        w_f = w_ref[0, :]                               # (N,) from (1, N)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS)
        o_ref[...] = (x / rms) * w_f                   # float32 output


_pallas_fn = pl.pallas_call(
    _kernel,
    jax.ShapeDtypeStruct((M, N), jnp.float32),    # float32 — cast outside
    grid=(M // BM, NUM_K),
    in_specs=[
        pl.BlockSpec((BM, BK), lambda m, k: (m, k)),   # a: (BM, BK) bf16
        pl.BlockSpec((BK, N),  lambda m, k: (k, 0)),   # b: (BK, N)  bf16, full N
        pl.BlockSpec((1,  N),  lambda m, k: (0, 0)),   # w: (1, N)   f32
    ],
    out_specs=pl.BlockSpec((BM, N), lambda m, k: (m, 0)),
    scratch_shapes=[pltpu.VMEM((BM, N), jnp.float32)],  # acc never touches HBM
)


def run(a: jnp.ndarray, b: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    a: (512, 768)  bfloat16
    b: (768, 768)  bfloat16
    w: (768,)      bfloat16  ← reshaped to (1,768) float32 internally
    returns: (512, 768) bfloat16  — result of matmul(a,b) normalized by RMSNorm(w)
    """
    return _pallas_fn(a, b, w.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


run = jax.jit(run)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
    w = jax.random.normal(key, (N,),   dtype=jnp.bfloat16)
    out = run(a, b, w).block_until_ready()

    # Reference: separate matmul + RMSNorm
    c  = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))
    rms = jnp.sqrt(jnp.mean(c ** 2, axis=-1, keepdims=True) + EPS)
    ref = ((c / rms) * w.astype(jnp.float32)).astype(jnp.bfloat16)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"kernel_010  shape=({M},{K}x{K},{N})+norm  max_err={err:.4e}  {'PASS' if err < 0.05 else 'FAIL'}")
