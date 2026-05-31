"""Smoke tests for MCP tool functions — called directly, not via MCP transport."""
import pytest
from kernel_factory import mcp_server


def test_solve_tile_config_matmul():
    result = mcp_server.solve_tile_config(
        op_type="matmul", M=512, N=512, K=256, tpu_version="v5e"
    )
    assert "block_m" in result
    assert result["block_m"] > 0
    assert result["vmem_utilization_pct"] <= 100.0


def test_solve_tile_config_rmsnorm():
    result = mcp_server.solve_tile_config(
        op_type="rmsnorm", M=256, N=512, K=512, tpu_version="v5e"
    )
    assert "block_m" in result


def test_solve_tile_config_bad_op():
    result = mcp_server.solve_tile_config(
        op_type="attention", M=512, N=512, K=256
    )
    assert "error" in result


def test_retrieve_template_static_fallback(tmp_path):
    result = mcp_server.retrieve_template(
        op_type="matmul", M=512, N=512, K=256,
        rag_path=str(tmp_path / ".lancedb"),  # not seeded
    )
    assert result["source"] == "static_fallback"
    assert "run_matmul" in result["template_code"]


def test_assemble_kernel_matmul(tmp_path):
    result = mcp_server.assemble_kernel(
        op_type="matmul", M=256, N=256, K=128, tpu_version="v5e",
        rag_path=str(tmp_path / ".lancedb"),
    )
    assert "assembled_code" in result
    assert "run_matmul" in result["assembled_code"]
    assert "error" not in result


def test_assemble_kernel_rmsnorm(tmp_path):
    result = mcp_server.assemble_kernel(
        op_type="rmsnorm", M=128, N=256, K=256, tpu_version="v5e",
        rag_path=str(tmp_path / ".lancedb"),
    )
    assert "run_rmsnorm" in result["assembled_code"]


def test_verify_kernel_matmul_passes():
    result = mcp_server.verify_kernel(
        op_type="matmul", M=256, N=256, K=128, tpu_version="v5e"
    )
    assert result["passed"] is True
    assert result["max_abs_error"] is not None


def test_verify_kernel_rmsnorm_passes():
    result = mcp_server.verify_kernel(
        op_type="rmsnorm", M=128, N=256, K=256, tpu_version="v5e"
    )
    assert result["passed"] is True


def test_mcp_server_importable_without_running():
    import kernel_factory.mcp_server as srv
    assert hasattr(srv, "solve_tile_config")
    assert hasattr(srv, "assemble_kernel")
    assert hasattr(srv, "verify_kernel")
    assert hasattr(srv, "retrieve_template")
