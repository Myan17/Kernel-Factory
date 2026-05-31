import pytest
from kernel_factory.schemas import LayerSpec, HardwareLimits, DType, KernelConfig
from kernel_factory.solver import TileSolver


def _hw() -> HardwareLimits:
    return HardwareLimits.for_v5e()


def test_solver_returns_config_for_matmul():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    config = TileSolver(_hw()).solve(spec)
    assert isinstance(config, KernelConfig)
    assert config.block_m > 0
    assert config.block_n > 0
    assert config.block_k > 0


def test_solver_vmem_within_budget():
    spec = LayerSpec(op_type="matmul", M=4096, N=4096, K=2048)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.total_vmem_estimate_bytes <= hw.vmem_budget_bytes


def test_solver_last_dim_aligned_to_vector_width():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_n % hw.vector_width == 0
    assert config.block_k % hw.vector_width == 0


def test_solver_second_dim_aligned_to_sublane():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_m % hw.sublane_width == 0


def test_solver_v4_small_matrices():
    spec = LayerSpec(op_type="matmul", M=64, N=64, K=64)
    config = TileSolver(HardwareLimits.for_v4()).solve(spec)
    assert config is not None


def test_solver_rmsnorm():
    spec = LayerSpec(op_type="rmsnorm", M=1024, N=4096, K=4096)
    config = TileSolver(_hw()).solve(spec)
    assert config.block_m > 0
    assert config.block_n > 0


def test_solver_vmem_utilization_fraction():
    spec = LayerSpec(op_type="matmul", M=2048, N=2048, K=1024)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert 0.0 < config.vmem_utilization_fraction <= 1.0


def test_solver_raises_for_unsupported_op():
    spec = LayerSpec(op_type="attention", M=512, N=512, K=512)
    with pytest.raises(ValueError, match="Unsupported op_type"):
        TileSolver(_hw()).solve(spec)


def test_solver_fused_matmul_rmsnorm():
    spec = LayerSpec(op_type="fused_matmul_rmsnorm", M=8192, N=768, K=768)
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_n == 768        # N is never tiled
    assert config.total_vmem_estimate_bytes <= hw.vmem_budget_bytes


def test_solver_flash_attention():
    spec = LayerSpec(
        op_type="flash_attention", M=2048, N=2048, K=64,
        seq_len=2048, num_heads=12, head_dim=64,
    )
    hw = _hw()
    config = TileSolver(hw).solve(spec)
    assert config.block_m > 0
    assert config.block_k > 0
    assert config.total_vmem_estimate_bytes <= hw.vmem_budget_bytes
