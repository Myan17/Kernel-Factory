"""All tests run in CPU interpret=True mode — no TPU required."""
import ast
import pytest
from kernel_factory.schemas import LayerSpec, HardwareLimits
from kernel_factory.pipeline import KernelPipeline, PipelineResult


def _hw() -> HardwareLimits:
    return HardwareLimits.for_v5e()


def test_pipeline_returns_result_for_matmul():
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    result = KernelPipeline(hw=_hw()).run(spec)
    assert isinstance(result, PipelineResult)
    assert result.kernel_config.block_m > 0
    assert "run_matmul" in result.assembled_code


def test_pipeline_returns_result_for_rmsnorm():
    spec = LayerSpec(op_type="rmsnorm", M=256, N=512, K=512)
    result = KernelPipeline(hw=_hw()).run(spec)
    assert isinstance(result, PipelineResult)
    assert "run_rmsnorm" in result.assembled_code


def test_pipeline_test_result_passes():
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    result = KernelPipeline(hw=_hw()).run(spec)
    assert result.test_result.passed is True


def test_pipeline_template_source_is_logged():
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    result = KernelPipeline(hw=_hw(), rag=None).run(spec)
    assert result.template_source in ("rag", "static_fallback")


def test_pipeline_assembled_code_is_valid_python():
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    result = KernelPipeline(hw=_hw()).run(spec)
    ast.parse(result.assembled_code)


def test_pipeline_raises_for_unsupported_op():
    spec = LayerSpec(op_type="attention", M=512, N=512, K=512)
    with pytest.raises((ValueError, RuntimeError)):
        KernelPipeline(hw=_hw()).run(spec)


def test_pipeline_with_sqlite_logging(tmp_path):
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    db = tmp_path / "results.db"
    result = KernelPipeline(hw=_hw(), db_path=db).run(spec)
    assert result.test_result.passed is True
    assert db.exists()


def test_pipeline_template_source_rag_when_seeded(tmp_path):
    from kernel_factory.rag import TemplateRAG
    rag = TemplateRAG(db_path=tmp_path / ".lancedb")
    rag.seed()
    spec = LayerSpec(op_type="matmul", M=256, N=256, K=128)
    result = KernelPipeline(hw=_hw(), rag=rag).run(spec)
    assert result.template_source == "rag"
    assert "run_matmul" in result.assembled_code
