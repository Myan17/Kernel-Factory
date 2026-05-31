"""
kernel_008 — GPT-2 small MatMul: FFN down projection  ★ BEATS XLA (+5%)
Verified on: Google Cloud TPU v5e (v5litepod-1, us-south1-a)
JAX version: 0.6.2
Benchmark:   XLA 0.130 ms / 18.58 TFLOPS  →  Pallas 0.124 ms / 19.56 TFLOPS (1.05x)

Shape: M=512, N=768, K=3072  (seq_len × 4×hidden → hidden)
Tiles: block_m=512, block_n=256, block_k=512  (VMEM ~1.25 MiB / 1.3% of budget)

Note: solver chose block_k=512 (6 K-tiles over K=3072). XLA's autotuner
picked a less efficient K-tiling for this asymmetric shape, giving us +5%.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M,  N,  K  = 512, 768, 3072
BM, BN, BK = 512, 256,  512
NUM_K       = K // BK   # 6


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
    jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
    grid=(M // BM, N // BN, NUM_K),
    in_specs=[
        pl.BlockSpec((BM, BK), lambda m, n, k: (m, k)),
        pl.BlockSpec((BK, BN), lambda m, n, k: (k, n)),
    ],
    out_specs=pl.BlockSpec((BM, BN), lambda m, n, k: (m, n)),
    scratch_shapes=[pltpu.VMEM((BM, BN), jnp.float32)],
)

run = jax.jit(_pallas_fn)
"""run(a, b) — a: (512, 3072) bf16, b: (3072, 768) bf16 → (512, 768) bf16"""


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
    out = run(a, b).block_until_ready()
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(jnp.bfloat16)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"kernel_008  shape=({M},{K}x{K},{N})  max_err={err:.4e}  {'PASS' if err < 0.05 else 'FAIL'}")
