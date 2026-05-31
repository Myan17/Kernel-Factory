import pytest
from pathlib import Path
from kernel_factory.schemas import LayerSpec, DType
from kernel_factory.rag import TemplateRAG


@pytest.fixture
def seeded_rag(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    rag.seed()
    return rag


def test_rag_seed_returns_count(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    n = rag.seed()
    assert n == 2  # matmul + rmsnorm


def test_rag_is_seeded_after_seed(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    assert not rag.is_seeded()
    rag.seed()
    assert rag.is_seeded()


def test_rag_seed_idempotent(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    rag.seed()
    n2 = rag.seed()          # second call without force
    assert n2 == 0           # skipped, returns 0


def test_rag_seed_force_reseeds(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    rag.seed()
    n2 = rag.seed(force=True)
    assert n2 == 2


def test_rag_retrieve_matmul(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    code = seeded_rag.retrieve(spec)
    assert "run_matmul" in code
    assert "matmul_kernel" in code


def test_rag_retrieve_rmsnorm(seeded_rag):
    spec = LayerSpec(op_type="rmsnorm", M=1024, N=4096, K=4096)
    code = seeded_rag.retrieve(spec)
    assert "run_rmsnorm" in code
    assert "rmsnorm_kernel" in code


def test_rag_retrieve_raises_if_not_seeded(tmp_path):
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    with pytest.raises(RuntimeError, match="not seeded"):
        rag.retrieve(spec)


def test_rag_retrieve_returns_string(seeded_rag):
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    result = seeded_rag.retrieve(spec)
    assert isinstance(result, str)
    assert len(result) > 100
