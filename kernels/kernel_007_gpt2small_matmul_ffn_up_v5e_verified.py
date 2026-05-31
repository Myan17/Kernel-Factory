"""
kernel_007 — GPT-2 small MatMul: FFN gate/up projection
Verified on: Google Cloud TPU v5e (v5litepod-1, us-south1-a)
JAX version: 0.6.2
Benchmark:   XLA 0.130 ms / 18.59 TFLOPS  →  Pallas 0.138 ms / 17.54 TFLOPS (0.94x)

Shape: M=512, N=3072, K=768  (seq_len × hidden → 4×hidden)
Tiles: block_m=512, block_n=512, block_k=256  (VMEM ~2.75 MiB / 2.9% of budget)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M,  N,   K  = 512, 3072, 768
BM, BN,  BK = 512,  512, 256
NUM_K        = K // BK   # 3


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
"""run(a, b) — a: (512, 768) bf16, b: (768, 3072) bf16 → (512, 3072) bf16"""


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K),  dtype=jnp.bfloat16)
    b = jax.random.normal(key, (K, N),  dtype=jnp.bfloat16)
    out = run(a, b).block_until_ready()
    ref = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)).astype(jnp.bfloat16)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"kernel_007  shape=({M},{K}x{K},{N})  max_err={err:.4e}  {'PASS' if err < 0.05 else 'FAIL'}")
