# Verified Pallas skeleton templates.
# Parameters substituted by Assembler — no Pallas logic is ever written from scratch.
# Substitution targets: {block_m} {block_n} {block_k} {M} {N} {K}
#                       {input_dtype} {output_dtype} {accumulator_dtype} {num_k_tiles}

MATMUL_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

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
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.{output_dtype}),
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
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.{output_dtype}),
        grid=(M // block_m, N // block_n),
        in_specs=in_specs,
        out_specs=out_specs,
    )(x, w)
'''

FUSED_MATMUL_RMSNORM_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

_EPS = 1e-6

def fused_matmul_rmsnorm_kernel(a_ref, b_ref, w_ref, o_ref, acc_ref):
    """
    Fused MatMul + RMSNorm.
    K is tiled (program_id 1). N is NOT tiled — the full output row
    accumulates in acc_ref so RMSNorm can see it in one pass.
    """
    @pl.when(pl.program_id(1) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.{accumulator_dtype}),
        b_ref[...].astype(jnp.{accumulator_dtype}),
        preferred_element_type=jnp.{accumulator_dtype},
    )

    @pl.when(pl.program_id(1) == {num_k_tiles} - 1)
    def _normalise():
        x = acc_ref[...]
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + _EPS)
        normed = x / rms
        o_ref[...] = (normed * w_ref[...].astype(jnp.{accumulator_dtype})).astype(
            jnp.{output_dtype}
        )


def run_fused_matmul_rmsnorm(
    a: jnp.ndarray, b: jnp.ndarray, w: jnp.ndarray
) -> jnp.ndarray:
    M, K = a.shape
    _, N = b.shape
    block_m, block_k = {block_m}, {block_k}
    num_k_tiles = {num_k_tiles}

    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda m, k: (m, k)),
        pl.BlockSpec((block_k, N),       lambda m, k: (k, 0)),
        pl.BlockSpec((N,),               lambda m, k: (0,)),
    ]
    out_specs    = pl.BlockSpec((block_m, N), lambda m, k: (m, 0))
    scratch_shapes = [pltpu.VMEM((block_m, N), jnp.{accumulator_dtype})]

    return pl.pallas_call(
        fused_matmul_rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.{output_dtype}),
        grid=(M // block_m, num_k_tiles),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )(a, b, w)
'''

FLASH_ATTENTION_TEMPLATE = '''\
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref,
    m_ref, l_ref,
):
    """
    Flash Attention forward pass.
    grid: (batch, num_heads, seq_q // block_q, seq_k // block_k).
    m_ref and l_ref are running max and normaliser (scratch).
    Score matrix never written to HBM.
    """
    scale = {scale}

    @pl.when(pl.program_id(3) == 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        o_ref[...] = jnp.zeros_like(o_ref)

    q = q_ref[...].astype(jnp.{accumulator_dtype})
    k = k_ref[...].astype(jnp.{accumulator_dtype})
    v = v_ref[...].astype(jnp.{accumulator_dtype})

    s = jnp.dot(q, k.T, preferred_element_type=jnp.{accumulator_dtype}) * scale
    m_new = jnp.maximum(m_ref[...], jnp.max(s, axis=-1, keepdims=True))
    p = jnp.exp(s - m_new)
    l_new = jnp.exp(m_ref[...] - m_new) * l_ref[...] + jnp.sum(p, axis=-1, keepdims=True)

    o_ref[...] = (
        jnp.exp(m_ref[...] - m_new) * o_ref[...]
        + jnp.dot(p, v, preferred_element_type=jnp.{accumulator_dtype})
    )
    m_ref[...] = m_new
    l_ref[...] = l_new

    @pl.when(pl.program_id(3) == {num_k_tiles} - 1)
    def _finalise():
        o_ref[...] = (o_ref[...] / l_ref[...]).astype(jnp.{output_dtype})


def run_flash_attention(
    q: jnp.ndarray,  # (batch, num_heads, seq_q, head_dim)
    k: jnp.ndarray,  # (batch, num_heads, seq_k, head_dim)
    v: jnp.ndarray,  # (batch, num_heads, seq_k, head_dim)
) -> jnp.ndarray:
    batch, num_heads, seq_q, head_dim = q.shape
    _,     _,         seq_k, _        = k.shape
    block_q, block_k = {block_m}, {block_k}
    num_k_tiles = seq_k // block_k
    scale = {scale}

    def q_idx(b, h, i, j):    return (b, h, i, 0)
    def kv_idx(b, h, i, j):   return (b, h, j, 0)

    in_specs = [
        pl.BlockSpec((1, 1, block_q, head_dim), q_idx),
        pl.BlockSpec((1, 1, block_k, head_dim), kv_idx),
        pl.BlockSpec((1, 1, block_k, head_dim), kv_idx),
    ]
    out_specs = pl.BlockSpec((1, 1, block_q, head_dim), q_idx)
    scratch_shapes = [
        pltpu.VMEM((1, 1, block_q, 1), jnp.{accumulator_dtype}),   # m
        pltpu.VMEM((1, 1, block_q, 1), jnp.{accumulator_dtype}),   # l
    ]

    return pl.pallas_call(
        flash_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((batch, num_heads, seq_q, head_dim), jnp.{output_dtype}),
        grid=(batch, num_heads, seq_q // block_q, num_k_tiles),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
        ),
    )(q, k, v)
'''

TEMPLATES: dict[str, str] = {
    "matmul":               MATMUL_TEMPLATE,
    "rmsnorm":              RMSNORM_TEMPLATE,
    "fused_matmul_rmsnorm": FUSED_MATMUL_RMSNORM_TEMPLATE,
    "flash_attention":      FLASH_ATTENTION_TEMPLATE,
}
