"""
kernel_006 — GPT-2 small MatMul: Attention QKV / Output projection
Verified on: Google Cloud TPU v5e (v5litepod-1, us-south1-a)
JAX version: 0.6.2
Benchmark:   XLA 0.108 ms / 5.60 TFLOPS  →  Pallas 0.121 ms / 5.00 TFLOPS (0.89x)

Shape: M=512, N=768, K=768  (seq_len × hidden → hidden)
Tiles: block_m=512, block_n=256, block_k=256  (VMEM ~1.25 MiB / 1.3% of budget)

This is the EXACT kernel that ran in benchmark_001. Do not rename the internal
functions or change tile sizes — the block specs are baked to these values.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ── Tile configuration (solver output for v5e, 512×768×768) ──────────────────
M,  N,  K  = 512, 768, 768
BM, BN, BK = 512, 256, 256
NUM_K       = K // BK   # 3


def _kernel(a_ref, b_ref, o_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == NUM_K - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(jnp.bfloat16)


_pallas_fn = pl.pallas_call(
    _kernel,
    jax.ShapeDtypeStruct((M, N), jnp.bfloat16),   # out_shape required in JAX 0.6.2
    grid=(M // BM, N // BN, NUM_K),
    in_specs=[
        pl.BlockSpec((BM, BK), lambda m, n, k: (m, k)),
        pl.BlockSpec((BK, BN), lambda m, n, k: (k, n)),
    ],
    out_specs=pl.BlockSpec((BM, BN), lambda m, n, k: (m, n)),
    scratch_shapes=[pltpu.VMEM((BM, BN), jnp.float32)],
)

run = jax.jit(_pallas_fn)
"""run(a, b) — a: (512, 768) bf16, b: (768, 768) bf16 → (512, 768) bf16"""


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
    out = run(a, b).block_until_ready()
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(jnp.bfloat16)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"kernel_006  shape=({M},{K}x{K},{N})  max_err={err:.4e}  {'PASS' if err < 0.05 else 'FAIL'}")
