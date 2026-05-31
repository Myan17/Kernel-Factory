"""Tests for ProductionRAG. Uses synthetic chunks — no real repo clone needed."""
import pytest
from pathlib import Path
from kernel_factory.rag import ProductionRAG
from kernel_factory.chunker import KernelChunk, KernelChunker
from kernel_factory.schemas import LayerSpec, DType


SAMPLE_MATMUL_CODE = '''
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    """Dense MatMul Pallas kernel for bfloat16 inputs."""
    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.float32),
        b_ref[...].astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )

def run_matmul(a, b):
    """Wrapper: calls pallas_call with matmul_kernel."""
    return pl.pallas_call(
        matmul_kernel,
        grid=(M // block_m, N // block_n, K // block_k),
        in_specs=[pl.BlockSpec((block_m, block_k), lambda m, n, k: (m, k))],
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n, k: (m, n)),
    )(a, b)
'''

SAMPLE_RMSNORM_CODE = '''
def rmsnorm_kernel(x_ref, w_ref, o_ref):
    """RMSNorm Pallas kernel."""
    x = x_ref[...].astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    o_ref[...] = ((x / rms) * w_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)

def run_rmsnorm(x, w):
    """Wrapper: calls pallas_call with rmsnorm_kernel."""
    return pl.pallas_call(
        rmsnorm_kernel,
        grid=(M // block_m, N // block_n),
        in_specs=[pl.BlockSpec((block_m, block_n), lambda m, n: (m, n))],
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m, n)),
    )(x, w)
'''

SAMPLE_ATTENTION_CODE = '''
def flash_attention_kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref):
    """Flash Attention TPU kernel with causal masking."""
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    scale = jnp.sqrt(jnp.array(q.shape[-1], dtype=jnp.float32))
    s = jnp.dot(q, k.T) / scale
    causal_mask = jnp.tril(jnp.ones_like(s))
    s = jnp.where(causal_mask, s, -1e9)
    p = jax.nn.softmax(s, axis=-1)
    o_ref[...] = jnp.dot(p, v_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
    m_ref[...] = jnp.max(s, axis=-1)
    l_ref[...] = jnp.sum(jnp.exp(s - m_ref[...][..., None]), axis=-1)
'''


def _make_chunks() -> list[KernelChunk]:
    chunker = KernelChunker()
    chunks = []
    chunks += chunker.chunk_file(SAMPLE_MATMUL_CODE, "ops/tpu/matmul.py", "jax-ml/jax", tier=1)
    chunks += chunker.chunk_file(SAMPLE_RMSNORM_CODE, "layers/normalizations.py", "AI-Hypercomputer/maxtext", tier=2)
    chunks += chunker.chunk_file(SAMPLE_ATTENTION_CODE, "ops/tpu/flash_attention.py", "jax-ml/jax", tier=1)
    return chunks


@pytest.fixture
def seeded_rag(tmp_path) -> ProductionRAG:
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    rag.upsert_chunks(_make_chunks())
    return rag


def test_rag_upsert_returns_count(tmp_path):
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    n = rag.upsert_chunks(_make_chunks())
    assert n >= 3


def test_rag_is_seeded_after_upsert(tmp_path):
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    assert not rag.is_seeded()
    rag.upsert_chunks(_make_chunks())
    assert rag.is_seeded()


def test_rag_upsert_is_idempotent(tmp_path):
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    n1 = rag.upsert_chunks(_make_chunks())
    n2 = rag.upsert_chunks(_make_chunks())
    assert n2 == 0   # all chunk_ids already exist


def test_rag_count_increases_with_upsert(tmp_path):
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    rag.upsert_chunks(_make_chunks())
    assert rag.count() >= 3


def test_rag_retrieve_matmul(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    results = seeded_rag.retrieve(spec, top_k=1)
    assert len(results) >= 1
    assert "matmul" in results[0]["chunk_text"].lower()


def test_rag_retrieve_rmsnorm(seeded_rag):
    spec = LayerSpec(op_type="rmsnorm", M=512, N=4096, K=4096)
    results = seeded_rag.retrieve(spec, top_k=1)
    assert len(results) >= 1
    top_text = results[0]["chunk_text"].lower()
    assert "norm" in top_text or "rms" in top_text


def test_rag_retrieve_one_returns_string(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    result = seeded_rag.retrieve_one(spec)
    assert isinstance(result, str)
    assert len(result) > 50


def test_rag_kernel_class_filter(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    results = seeded_rag.retrieve(spec, top_k=3, kernel_class_filter="attention")
    for r in results:
        assert r["kernel_class"] == "attention"


def test_rag_result_has_required_fields(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    results = seeded_rag.retrieve(spec, top_k=1)
    required = {"chunk_id", "source_repo", "source_file", "function_name",
                "kernel_class", "op_type", "chunk_text", "tier"}
    assert required.issubset(results[0].keys())
    assert "vector" not in results[0]   # vector must NOT be returned


def test_rag_stats(seeded_rag):
    stats = seeded_rag.stats()
    assert "total_chunks" in stats
    assert stats["total_chunks"] >= 3
    assert "by_kernel_class" in stats
    assert "by_source_repo" in stats


def test_rag_fallback_when_empty(tmp_path):
    rag = ProductionRAG(db_path=tmp_path / ".lancedb")
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    result = rag.retrieve_one(spec)
    assert isinstance(result, str)
    assert len(result) > 0   # should return static template fallback
