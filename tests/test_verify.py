"""
Verification harness tests — CPU interpret=True mode only, no TPU required.
"""
import sqlite3
import pytest
from kernel_factory.schemas import LayerSpec, HardwareLimits, DType
from kernel_factory.solver import TileSolver
from kernel_factory.verify import VerificationGate, VerifyMode


def _solve(op_type, M, N, K, **kwargs):
    spec = LayerSpec(op_type=op_type, M=M, N=N, K=K, **kwargs)
    config = TileSolver(HardwareLimits.for_v5e()).solve(spec)
    return spec, config


def test_matmul_cpu_interpret_passes():
    spec, config = _solve("matmul", 64, 64, 64)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.passed, result.error_trace


def test_rmsnorm_cpu_interpret_passes():
    spec, config = _solve("rmsnorm", 64, 128, 128)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.passed, result.error_trace


def test_result_has_latency():
    spec, config = _solve("matmul", 64, 64, 64)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.execution_latency_ms is not None
    assert result.execution_latency_ms > 0


def test_result_has_max_abs_error():
    spec, config = _solve("matmul", 64, 64, 64)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.max_abs_error is not None
    assert result.max_abs_error >= 0.0


def test_logged_to_sqlite(tmp_path):
    spec, config = _solve("matmul", 64, 64, 64)
    db = tmp_path / "results.db"
    VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET, db_path=db).run()
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT op_type, passed, block_m FROM kernel_results").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "matmul"
    assert rows[0][2] == config.block_m


def test_multiple_runs_accumulate_in_db(tmp_path):
    db = tmp_path / "results.db"
    for _ in range(3):
        spec, config = _solve("matmul", 64, 64, 64)
        VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET, db_path=db).run()
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM kernel_results").fetchone()[0]
    conn.close()
    assert count == 3


def test_tpu_mode_returns_informative_error():
    spec, config = _solve("matmul", 64, 64, 64)
    result = VerificationGate(spec, config, mode=VerifyMode.TPU_COMPILE).run()
    assert not result.passed
    assert "TPU VM" in result.error_trace


def test_unknown_op_returns_error():
    spec = LayerSpec(op_type="matmul", M=64, N=64, K=64)
    config = TileSolver(HardwareLimits.for_v5e()).solve(spec)
    # Manually patch op_type to something unsupported
    spec2 = spec.model_copy(update={"op_type": "attention"})
    result = VerificationGate(spec2, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert not result.passed
    assert "attention" in result.error_trace


def test_fused_matmul_rmsnorm_cpu_interpret_passes():
    spec = LayerSpec(op_type="fused_matmul_rmsnorm", M=64, N=128, K=128)
    config = TileSolver(HardwareLimits.for_v5e()).solve(spec)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.passed, result.error_trace


def test_flash_attention_cpu_interpret_passes():
    spec = LayerSpec(
        op_type="flash_attention", M=128, N=128, K=64,
        seq_len=128, num_heads=1, head_dim=64,
    )
    config = TileSolver(HardwareLimits.for_v5e()).solve(spec)
    result = VerificationGate(spec, config, mode=VerifyMode.CPU_INTERPRET).run()
    assert result.passed, result.error_trace
