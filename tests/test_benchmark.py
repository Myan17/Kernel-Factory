"""Benchmark harness tests — CPU interpret=True mode only, no TPU required.

Real on-chip timings only mean something on a TPU; these tests validate the
harness *logic*: it runs the full pipeline, builds baseline + custom kernels,
compares them for correctness, and produces well-formed results + a table.
"""
import pytest

from kernel_factory.schemas import HardwareLimits, LayerSpec
from kernel_factory.benchmark import (
    BenchmarkResult,
    benchmark_one,
    run_benchmark,
    default_shapes,
    format_table,
)


def _hw() -> HardwareLimits:
    return HardwareLimits.for_v5e()


def test_benchmark_one_matmul_returns_result():
    spec = LayerSpec(op_type="matmul", M=128, N=128, K=128)
    r = benchmark_one(spec, _hw(), interpret=True, iters=2)
    assert isinstance(r, BenchmarkResult)
    assert r.op_type == "matmul"
    assert r.baseline_ms > 0
    assert r.custom_ms > 0
    assert r.speedup == pytest.approx(r.baseline_ms / r.custom_ms, rel=1e-6)


def test_benchmark_one_rmsnorm_returns_result():
    spec = LayerSpec(op_type="rmsnorm", M=128, N=128, K=128)
    r = benchmark_one(spec, _hw(), interpret=True, iters=2)
    assert isinstance(r, BenchmarkResult)
    assert r.op_type == "rmsnorm"
    assert r.baseline_ms > 0
    assert r.custom_ms > 0


def test_benchmark_matmul_is_numerically_correct():
    """Custom kernel must match the XLA baseline within tolerance."""
    spec = LayerSpec(op_type="matmul", M=128, N=128, K=128)
    r = benchmark_one(spec, _hw(), interpret=True, iters=2)
    assert r.passed is True, f"max_abs_error={r.max_abs_error}"
    assert r.max_abs_error is not None
    assert r.max_abs_error >= 0.0


def test_benchmark_rmsnorm_is_numerically_correct():
    """RMSNorm reduces over the full feature dim; custom must match baseline."""
    spec = LayerSpec(op_type="rmsnorm", M=128, N=256, K=256)
    r = benchmark_one(spec, _hw(), interpret=True, iters=2)
    assert r.passed is True, f"max_abs_error={r.max_abs_error}"


def test_benchmark_records_throughput_unit():
    mm = benchmark_one(LayerSpec(op_type="matmul", M=128, N=128, K=128),
                       _hw(), interpret=True, iters=2)
    rn = benchmark_one(LayerSpec(op_type="rmsnorm", M=128, N=128, K=128),
                       _hw(), interpret=True, iters=2)
    assert mm.throughput_unit == "TFLOP/s"
    assert mm.throughput > 0
    assert rn.throughput_unit == "GB/s"
    assert rn.throughput > 0


def test_benchmark_runs_full_pipeline():
    """benchmark_one drives KernelPipeline.run — the whole factory, not a shortcut."""
    spec = LayerSpec(op_type="matmul", M=128, N=128, K=128)
    r = benchmark_one(spec, _hw(), interpret=True, iters=2)
    assert r.pipeline_passed is True
    assert r.block_m > 0 and r.block_n > 0 and r.block_k > 0


def test_run_benchmark_returns_one_result_per_spec():
    specs = [
        LayerSpec(op_type="matmul", M=128, N=128, K=128),
        LayerSpec(op_type="rmsnorm", M=128, N=128, K=128),
    ]
    results = run_benchmark(specs, _hw(), interpret=True, iters=2)
    assert len(results) == len(specs)
    assert all(isinstance(r, BenchmarkResult) for r in results)


def test_default_shapes_are_valid():
    shapes = default_shapes()
    assert len(shapes) > 0
    for s in shapes:
        assert s.op_type in ("matmul", "rmsnorm")
        # Last dim (N) must be a multiple of the 128-lane vector width.
        assert s.N % 128 == 0


def test_format_table_contains_ops_and_speedup():
    specs = [
        LayerSpec(op_type="matmul", M=128, N=128, K=128),
        LayerSpec(op_type="rmsnorm", M=128, N=128, K=128),
    ]
    results = run_benchmark(specs, _hw(), interpret=True, iters=2)
    table = format_table(results)
    assert "matmul" in table
    assert "rmsnorm" in table
    assert "speedup" in table.lower()
