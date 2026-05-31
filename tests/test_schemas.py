import pytest
from kernel_factory.schemas import (
    DType, LayerSpec, HardwareLimits, KernelConfig, TestResult
)


def test_hardware_limits_v5e():
    hw = HardwareLimits.for_v5e()
    assert hw.vmem_bytes == 16 * 1024 * 1024
    assert hw.vector_width == 128
    assert hw.sublane_width == 8


def test_hardware_limits_v4():
    hw = HardwareLimits.for_v4()
    assert hw.vmem_bytes == 16 * 1024 * 1024


def test_hardware_limits_v6e():
    hw = HardwareLimits.for_v6e()
    assert hw.vmem_bytes == 16 * 1024 * 1024
    assert hw.tpu_version == "v6e"


def test_vmem_budget_is_75_percent():
    hw = HardwareLimits.for_v5e()
    assert hw.vmem_budget_bytes == int(16 * 1024 * 1024 * 0.75)


def test_layer_spec_defaults():
    spec = LayerSpec(op_type="matmul", M=1024, N=1024, K=512)
    assert spec.input_dtype == DType.BFLOAT16
    assert spec.accumulator_dtype == DType.FLOAT32


def test_dtype_itemsize():
    assert DType.FLOAT32.itemsize == 4
    assert DType.BFLOAT16.itemsize == 2
    assert DType.INT8.itemsize == 1


def test_kernel_config_fields():
    config = KernelConfig(
        block_m=128, block_n=128, block_k=128,
        stages=1,
        input_dtype=DType.BFLOAT16,
        output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=2 * 1024 * 1024,
        vmem_utilization_fraction=0.015,
    )
    assert config.block_m == 128


def test_test_result_pass():
    config = KernelConfig(
        block_m=64, block_n=64, block_k=64, stages=1,
        input_dtype=DType.BFLOAT16, output_dtype=DType.BFLOAT16,
        accumulator_dtype=DType.FLOAT32,
        total_vmem_estimate_bytes=512 * 1024,
        vmem_utilization_fraction=0.004,
    )
    spec = LayerSpec(op_type="matmul", M=512, N=512, K=256)
    result = TestResult(kernel_config=config, layer_spec=spec, passed=True)
    assert result.passed is True
    assert result.error_trace is None
