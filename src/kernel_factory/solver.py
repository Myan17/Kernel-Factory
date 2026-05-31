from __future__ import annotations
from kernel_factory.schemas import DType, HardwareLimits, KernelConfig, LayerSpec

_CANDIDATE_POWERS = [16, 32, 64, 128, 256, 512]


# ── VMEM estimators ────────────────────────────────────────────────────────────

def _vmem_matmul(bm: int, bn: int, bk: int, spec: LayerSpec) -> int:
    item = spec.input_dtype.itemsize
    acc_item = spec.accumulator_dtype.itemsize
    a_tile   = 2 * bm * bk * item      # double-buffered A tiles
    b_tile   = 2 * bk * bn * item      # double-buffered B tiles
    out_tile = bm * bn * spec.output_dtype.itemsize
    # Mosaic pipelines the accumulator with stages=2, effectively double-buffering it.
    acc_tile = 2 * bm * bn * acc_item
    return a_tile + b_tile + out_tile + acc_tile


def _vmem_rmsnorm(bm: int, bn: int, spec: LayerSpec) -> int:
    ib = spec.input_dtype.itemsize
    ob = spec.output_dtype.itemsize
    ab = spec.accumulator_dtype.itemsize
    # input tile + weight vector + output tile + per-row accumulator
    return bm * bn * ib + bn * ib + bm * bn * ob + bm * ab


def _vmem_fused_matmul_rmsnorm(bm: int, bk: int, spec: LayerSpec) -> int:
    """
    Fused kernel keeps full N rows in VMEM for normalisation.
    A is double-buffered over K. B is double-buffered over K.
    Weight is broadcast to (bm, N) 2D to satisfy Mosaic tiling constraints.
    """
    N = spec.N
    item = spec.input_dtype.itemsize
    acc_item = spec.accumulator_dtype.itemsize
    a_tile   = 2 * bm * bk * item      # double-buffered
    b_tile   = 2 * bk * N  * item      # double-buffered; full N width
    acc      = bm * N * acc_item        # full output row accumulator
    w_tile   = bm * N * item            # weight broadcast to (bm, N)
    out_tile = bm * N * spec.output_dtype.itemsize
    return a_tile + b_tile + acc + w_tile + out_tile


def _vmem_flash_attention(bq: int, bk: int, spec: LayerSpec) -> int:
    """
    Q tile, K tile, V tile: each bq/bk × head_dim (double-buffered for K).
    Score tile S: bq × bk (float32, in VMEM).
    Output O: bq × head_dim (float32 accumulator).
    Running stats m, l: bq (float32 each).
    """
    d = spec.head_dim
    item = spec.input_dtype.itemsize
    acc_item = spec.accumulator_dtype.itemsize
    q_tile  = bq * d * item
    kv_tile = 2 * bk * d * item        # double-buffered K and V
    s_tile  = bq * bk * acc_item       # score matrix (float32)
    o_tile  = bq * d * acc_item        # output accumulator
    stats   = 2 * bq * acc_item        # m and l vectors
    return q_tile + kv_tile + s_tile + o_tile + stats


def _aligned(v: int, alignment: int) -> bool:
    return v % alignment == 0


# ── Solver ────────────────────────────────────────────────────────────────────

class TileSolver:
    def __init__(self, hw: HardwareLimits):
        self.hw = hw

    def solve(self, spec: LayerSpec) -> KernelConfig:
        if spec.op_type == "matmul":
            return self._solve_matmul(spec)
        elif spec.op_type == "rmsnorm":
            return self._solve_rmsnorm(spec)
        elif spec.op_type == "fused_matmul_rmsnorm":
            return self._solve_fused_matmul_rmsnorm(spec)
        elif spec.op_type == "flash_attention":
            return self._solve_flash_attention(spec)
        else:
            raise ValueError(f"Unsupported op_type: '{spec.op_type}'")

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
                            stages=2,
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
                        stages=2,
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

    def _solve_fused_matmul_rmsnorm(self, spec: LayerSpec) -> KernelConfig:
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1

        for bm in reversed(self._candidates(spec.M, hw.sublane_width)):
            for bk in reversed(self._candidates(spec.K, hw.vector_width)):
                vmem = _vmem_fused_matmul_rmsnorm(bm, bk, spec)
                if vmem > budget:
                    continue
                score = bm * bk
                if score > best_score:
                    best_score = score
                    best = KernelConfig(
                        block_m=bm,
                        block_n=spec.N,   # full N — not tiled
                        block_k=bk,
                        stages=2,
                        input_dtype=spec.input_dtype,
                        output_dtype=spec.output_dtype,
                        accumulator_dtype=spec.accumulator_dtype,
                        total_vmem_estimate_bytes=vmem,
                        vmem_utilization_fraction=vmem / hw.vmem_bytes,
                    )

        if best is None:
            raise RuntimeError(
                f"No valid tile found for fused_matmul_rmsnorm {spec} "
                f"within {budget} bytes VMEM budget"
            )
        return best

    def _solve_flash_attention(self, spec: LayerSpec) -> KernelConfig:
        assert spec.seq_len is not None, "seq_len required for flash_attention"
        assert spec.head_dim is not None, "head_dim required for flash_attention"
        hw = self.hw
        budget = hw.vmem_budget_bytes
        best: KernelConfig | None = None
        best_score = -1

        for bq in reversed(self._candidates(spec.seq_len, hw.sublane_width)):
            for bk in reversed(self._candidates(spec.seq_len, hw.vector_width)):
                vmem = _vmem_flash_attention(bq, bk, spec)
                if vmem > budget:
                    continue
                score = bq * bk
                if score > best_score:
                    best_score = score
                    best = KernelConfig(
                        block_m=bq,           # seq_q tile
                        block_n=spec.seq_len,
                        block_k=bk,           # seq_k tile
                        stages=2,
                        input_dtype=spec.input_dtype,
                        output_dtype=spec.output_dtype,
                        accumulator_dtype=spec.accumulator_dtype,
                        total_vmem_estimate_bytes=vmem,
                        vmem_utilization_fraction=vmem / hw.vmem_bytes,
                    )

        if best is None:
            raise RuntimeError(
                f"No flash_attention tile found for {spec} within {budget} bytes"
            )
        return best
