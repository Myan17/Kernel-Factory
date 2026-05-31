import pytest
from kernel_factory.chunker import KernelChunker, KernelChunk

SAMPLE_MATMUL = '''
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    """Dense MatMul Pallas kernel."""
    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
    )

def run_matmul(a, b):
    """Wrapper that calls pallas_call."""
    return pl.pallas_call(
        matmul_kernel,
        grid=(8, 8, 4),
        in_specs=[pl.BlockSpec((128, 128), lambda m, n, k: (m, k))],
        out_specs=pl.BlockSpec((128, 128), lambda m, n, k: (m, n)),
    )(a, b)

def _helper():
    pass
'''

SAMPLE_ATTENTION = '''
def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref, m_ref, l_ref
):
    """Flash Attention TPU kernel with online softmax."""
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    s = jnp.dot(q, k.T)
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - m[..., None])
    l = jnp.sum(p, axis=-1)
    o_ref[...] = (p @ v_ref[...]) / l[..., None]
    m_ref[...] = m
    l_ref[...] = l
'''


def test_chunker_extracts_functions():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "test/repo", tier=1)
    names = [c.function_name for c in chunks]
    assert "matmul_kernel" in names
    assert "run_matmul" in names


def test_chunker_skips_tiny_functions():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "test/repo", tier=1)
    names = [c.function_name for c in chunks]
    assert "_helper" not in names   # 2 lines — below threshold


def test_chunker_infers_matmul_class():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "test/repo", tier=1)
    matmul_chunk = next(c for c in chunks if c.function_name == "matmul_kernel")
    assert matmul_chunk.kernel_class == "matmul"
    assert matmul_chunk.op_type == "matmul"


def test_chunker_infers_attention_class():
    chunks = KernelChunker().chunk_file(SAMPLE_ATTENTION, "attn.py", "test/repo", tier=1)
    assert len(chunks) == 1
    assert chunks[0].kernel_class == "attention"


def test_chunker_tags_vmem():
    code = '''
def kernel_with_vmem(x_ref, o_ref):
    """Uses VMEM scratch."""
    scratch = pltpu.VMEM((128, 128), jnp.float32)
    o_ref[...] = x_ref[...]
'''
    chunks = KernelChunker().chunk_file(code, "f.py", "r", tier=1)
    assert "vmem" in chunks[0].tags


def test_chunker_chunk_id_is_deterministic():
    chunks1 = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "repo", tier=1)
    chunks2 = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "repo", tier=1)
    ids1 = {c.chunk_id for c in chunks1}
    ids2 = {c.chunk_id for c in chunks2}
    assert ids1 == ids2


def test_chunker_handles_syntax_error():
    bad_code = "def broken(\n    # unclosed"
    chunks = KernelChunker().chunk_file(bad_code, "bad.py", "repo", tier=1)
    assert chunks == []


def test_chunker_preserves_line_numbers():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "repo", tier=1)
    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start


def test_chunker_attaches_source_repo():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "jax-ml/jax", tier=1)
    assert all(c.source_repo == "jax-ml/jax" for c in chunks)


def test_chunker_attaches_tier():
    chunks = KernelChunker().chunk_file(SAMPLE_MATMUL, "test.py", "repo", tier=2)
    assert all(c.tier == 2 for c in chunks)
