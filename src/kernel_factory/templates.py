# Verified Pallas skeleton templates.
# Parameters substituted by Assembler — no Pallas logic is ever written from scratch.
# Substitution targets: {block_m} {block_n} {block_k} {M} {N} {K}
#                       {input_dtype} {output_dtype} {accumulator_dtype} {num_k_tiles}

MATMUL_TEMPLATE = '''\
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
        a_ref[...].astype(jnp.{accumulator_dtype}),
        b_ref[...].astype(jnp.{accumulator_dtype}),
        preferred_element_type=jnp.{accumulator_dtype},
    )

    @pl.when(pl.program_id(2) == {num_k_tiles} - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(jnp.{output_dtype})


def run_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k = {block_m}, {block_n}, {block_k}
    num_k_tiles = {num_k_tiles}

    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda m, n, k: (m, k)),
        pl.BlockSpec((block_k, block_n), lambda m, n, k: (k, n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n, k: (m, n))
    scratch_shapes = [pltpu.VMEM((block_m, block_n), jnp.{accumulator_dtype})]

    return pl.pallas_call(
        matmul_kernel,
        grid=(M // block_m, N // block_n, num_k_tiles),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, b)
'''

RMSNORM_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

_EPS = 1e-6

def rmsnorm_kernel(x_ref, w_ref, o_ref):
    """RMSNorm Pallas kernel — template-generated, do not edit by hand."""
    x = x_ref[...].astype(jnp.{accumulator_dtype})
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + _EPS)
    o_ref[...] = ((x / rms) * w_ref[...].astype(jnp.{accumulator_dtype})).astype(jnp.{output_dtype})


def run_rmsnorm(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    M, N = x.shape
    block_m, block_n = {block_m}, {block_n}

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
'''

TEMPLATES: dict[str, str] = {
    "matmul": MATMUL_TEMPLATE,
    "rmsnorm": RMSNORM_TEMPLATE,
}
