import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.ops.tpu as pltpu

def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    """Dense MatMul Pallas kernel — template-generated, do not edit by hand."""
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == 1 - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(jnp.bfloat16)


def run_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k = 512, 768, 768
    num_k_tiles = 1

    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda m, n, k: (m, k)),
        pl.BlockSpec((block_k, block_n), lambda m, n, k: (k, n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n, k: (m, n))
    scratch_shapes = [pltpu.VMEM((block_m, block_n), jnp.float32)]

    return pl.pallas_call(
        matmul_kernel,
        grid=(M // block_m, N // block_n, num_k_tiles),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
    )(a, b)
