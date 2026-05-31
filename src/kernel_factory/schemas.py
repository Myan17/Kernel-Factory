from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DType(str, Enum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"

    @property
    def itemsize(self) -> int:
        return {"float32": 4, "bfloat16": 2, "int8": 1}[self.value]


class LayerSpec(BaseModel):
    op_type: str  # "matmul" | "rmsnorm" | "fused_matmul_rmsnorm" | "flash_attention"
    M: int
    N: int
    K: int
    input_dtype: DType = DType.BFLOAT16
    output_dtype: DType = DType.BFLOAT16
    accumulator_dtype: DType = DType.FLOAT32
    batch_size: Optional[int] = None
    # Attention-specific fields (None for non-attention ops)
    seq_len: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None


class HardwareLimits(BaseModel):
    tpu_version: str
    vmem_bytes: int
    hbm_bandwidth_gbps: float
    vector_width: int = 128   # last dim must be a multiple of this
    sublane_width: int = 8    # second-to-last dim must be a multiple of this
    max_tiles_per_dim: int = 2048
    vmem_safety_fraction: float = Field(default=0.75, ge=0.0, le=1.0)

    @classmethod
    def for_v5e(cls) -> HardwareLimits:
        return cls(
            tpu_version="v5e",
            vmem_bytes=128 * 1024 * 1024,  # 128 MiB per TensorCore
            hbm_bandwidth_gbps=819.2,
        )

    @classmethod
    def for_v4(cls) -> HardwareLimits:
        return cls(
            tpu_version="v4",
            vmem_bytes=16 * 1024 * 1024,  # 16 MiB per TensorCore
            hbm_bandwidth_gbps=614.4,
        )

    @classmethod
    def for_v6e(cls) -> HardwareLimits:
        return cls(
            tpu_version="v6e",
            vmem_bytes=128 * 1024 * 1024,  # 128 MiB per TensorCore
            hbm_bandwidth_gbps=1638.4,
        )

    @property
    def vmem_budget_bytes(self) -> int:
        """Usable VMEM after safety margin (default 75% of total)."""
        return int(self.vmem_bytes * self.vmem_safety_fraction)


class KernelConfig(BaseModel):
    block_m: int
    block_n: int
    block_k: int
    stages: int = 2
    input_dtype: DType
    output_dtype: DType
    accumulator_dtype: DType
    total_vmem_estimate_bytes: int
    vmem_utilization_fraction: float


class TestResult(BaseModel):
    kernel_config: KernelConfig
    layer_spec: LayerSpec
    passed: bool
    max_abs_error: Optional[float] = None
    compile_time_ms: Optional[float] = None
    execution_latency_ms: Optional[float] = None
    error_trace: Optional[str] = None
    tpu_version: str = "unknown"
