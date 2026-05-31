import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

_EPS = 1e-6

def rmsnorm_kernel(x_ref, w_ref, o_ref):
    """RMSNorm Pallas kernel — template-generated, do not edit by hand."""
    x = x_ref[...].astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + _EPS)
    o_ref[...] = ((x / rms) * w_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)


def run_rmsnorm(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    M, N = x.shape
    block_m, block_n = 512, 768

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
