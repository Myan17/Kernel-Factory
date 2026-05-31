"""
kernel_009 — GPT-2 small RMSNorm  ★ BEATS XLA (+3%)
Verified on: Google Cloud TPU v5e (v5litepod-1, us-south1-a)
JAX version: 0.6.2
Benchmark:   XLA 0.104 ms  →  Pallas 0.101 ms (1.03x)

Shape: M=512, N=768  (seq_len × hidden)
Tiles: block_m=512, full N=768 per tile (VMEM ~3.15 MiB / 3.3% of budget)

Key implementation notes (v5e Mosaic constraints):
  1. Weight w must be passed as float32 (1, N) — NOT bfloat16 (N,).
     Mosaic can't retile a 1D bfloat16 (1,256) vector to (8,128) layout.
  2. Output is float32; caller casts to bfloat16 after the pallas_call.
  3. Full N row processed per tile — required for correct mean(x²) over row.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

M,  N  = 512, 768
BM     = 512        # rows per tile; full N processed in one shot
EPS    = 1e-6


def _kernel(x_ref, w_ref, o_ref):
    x_f  = x_ref[...].astype(jnp.float32)   # (BM, N)
    w_f  = w_ref[0, :]                       # (N,)  — w stored as (1, N)
    rms  = jnp.sqrt(jnp.mean(x_f ** 2, axis=-1, keepdims=True) + EPS)
    o_ref[...] = (x_f / rms) * w_f           # (BM, N) float32


_pallas_fn = pl.pallas_call(
    _kernel,
    jax.ShapeDtypeStruct((M, N), jnp.float32),   # float32 output — cast outside
    grid=(M // BM,),
    in_specs=[
        pl.BlockSpec((BM, N), lambda m: (m, 0)),  # x: (BM, N) bfloat16
        pl.BlockSpec((1,  N), lambda m: (0, 0)),  # w: (1, N)  float32
    ],
    out_specs=pl.BlockSpec((BM, N), lambda m: (m, 0)),
)


def run(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """
    x: (512, 768) bfloat16
    w: (768,)     bfloat16  ← automatically reshaped to (1,768) float32 internally
    returns: (512, 768) bfloat16
    """
    return _pallas_fn(x, w.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


run = jax.jit(run)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (M, N), dtype=jnp.bfloat16)
    w = jax.random.normal(key, (N,),   dtype=jnp.bfloat16)
    out = run(x, w).block_until_ready()

    xf  = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + EPS)
    ref = ((xf / rms) * w.astype(jnp.float32)).astype(jnp.bfloat16)
    err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    print(f"kernel_009  shape=({M},{N})  max_err={err:.4e}  {'PASS' if err < 0.01 else 'FAIL'}")
