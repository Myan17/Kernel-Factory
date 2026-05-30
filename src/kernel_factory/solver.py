from __future__ import annotations
from kernel_factory.schemas import DType, HardwareLimits, KernelConfig, LayerSpec

_CANDIDATE_POWERS = [16, 32, 64, 128, 256, 512]


def _vmem_matmul(bm: int, bn: int, bk: int, spec: LayerSpec) -> int:
    ib = spec.input_dtype.itemsize
    ob = spec.output_dtype.itemsize
    ab = spec.accumulator_dtype.itemsize
    return bm * bk * ib + bk * bn * ib + bm * bn * ob + bm * bn * ab


def _vmem_rmsnorm(bm: int, bn: int, spec: LayerSpec) -> int:
    ib = spec.input_dtype.itemsize
    ob = spec.output_dtype.itemsize
    ab = spec.accumulator_dtype.itemsize
    # input tile + weight vector + output tile + per-row accumulator
    return bm * bn * ib + bn * ib + bm * bn * ob + bm * ab


def _aligned(v: int, alignment: int) -> bool:
    return v % alignment == 0


class TileSolver:
    def __init__(self, hw: HardwareLimits):
        self.hw = hw

    def solve(self, spec: LayerSpec) -> KernelConfig:
        if spec.op_type == "matmul":
            return self._solve_matmul(spec)
        elif spec.op_type == "rmsnorm":
            return self._solve_rmsnorm(spec)
        raise ValueError(f"Unsupported op_type: {spec.op_type}")

    def _candidates(self, dim: int, must_align_to: int) -> list[int]:
        # A block equal to the full array dimension is always valid (TPU spec exception).
        result = [p for p in _CANDIDATE_POWERS if p <= dim and _aligned(p, must_align_to)]
        if dim not in result:
            result.append(dim)
        return sorted(result)

    def _solve_matmul(self, spec: LayerSpec) -> KernelConfig:
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1

        for bm in reversed(self._candidates(spec.M, hw.sublane_width)):
            for bn in reversed(self._candidates(spec.N, hw.vector_width)):
                for bk in reversed(self._candidates(spec.K, hw.vector_width)):
                    vmem = _vmem_matmul(bm, bn, bk, spec)
                    if vmem > budget:
                        continue
                    score = bm * bn * bk
                    if score > best_score:
                        best_score = score
                        best = KernelConfig(
                            block_m=bm, block_n=bn, block_k=bk,
                            stages=1,
                            input_dtype=spec.input_dtype,
                            output_dtype=spec.output_dtype,
                            accumulator_dtype=spec.accumulator_dtype,
                            total_vmem_estimate_bytes=vmem,
                            vmem_utilization_fraction=vmem / hw.vmem_bytes,
                        )

        if best is None:
            raise RuntimeError(
                f"No valid tile found for {spec} within {budget} bytes VMEM budget"
            )
        return best

    def _solve_rmsnorm(self, spec: LayerSpec) -> KernelConfig:
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1

        for bm in reversed(self._candidates(spec.M, hw.sublane_width)):
            for bn in reversed(self._candidates(spec.N, hw.vector_width)):
                vmem = _vmem_rmsnorm(bm, bn, spec)
                if vmem > budget:
                    continue
                score = bm * bn
                if score > best_score:
                    best_score = score
                    best = KernelConfig(
                        block_m=bm, block_n=bn, block_k=spec.K,
                        stages=1,
                        input_dtype=spec.input_dtype,
                        output_dtype=spec.output_dtype,
                        accumulator_dtype=spec.accumulator_dtype,
                        total_vmem_estimate_bytes=vmem,
                        vmem_utilization_fraction=vmem / hw.vmem_bytes,
                    )

        if best is None:
            raise RuntimeError(
                f"No valid tile found for RMSNorm {spec} within {budget} bytes"
            )
        return best
