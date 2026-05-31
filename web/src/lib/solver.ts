// TileSolver — ported 1:1 from kernel_factory/solver.py.
//
// The pipeline's intellectual core: given (M, N, K), a dtype, and a TPU's
// hardware limits it COMPUTES the largest valid tile (block_m, block_n,
// block_k) with no search — pure constraint math. This is the real algorithm,
// so the browser demo is faithful, not faked. The numbers come from hardware
// math, not a guess.

import {
  DTYPE_ITEMSIZE,
  type DType,
  type HardwareLimits,
  vmemBudgetBytes,
} from "./hardware";

export type OpType = "matmul" | "rmsnorm";

export interface LayerSpec {
  opType: OpType;
  M: number;
  N: number;
  K: number;
  inputDtype: DType;
  outputDtype: DType;
  accumulatorDtype: DType;
}

export interface KernelConfig {
  blockM: number;
  blockN: number;
  blockK: number;
  stages: number;
  inputDtype: DType;
  outputDtype: DType;
  accumulatorDtype: DType;
  totalVmemEstimateBytes: number;
  vmemUtilizationFraction: number;
  numKTiles: number;
}

// ── candidate-elimination metadata (for the live-demo visualization) ──────────

export type Axis = "m" | "n" | "k";

export interface CandidateChip {
  value: number;
  aligned: boolean; // passes the %alignment rule
  withinDim: boolean; // value <= dimension
  isFullDim: boolean; // the full-array-dimension exception
  kept: boolean; // survives into the solver's candidate set
}

export interface AxisBreakdown {
  axis: Axis;
  dim: number;
  alignment: number; // sublane_width (m) or vector_width (n, k)
  alignmentLabel: string;
  chips: CandidateChip[];
}

export interface SolveResult {
  ok: boolean;
  error?: string;
  config?: KernelConfig;
  axes: AxisBreakdown[];
  budgetBytes: number;
  vmemBytes: number;
  // combo accounting over the aligned candidate space
  totalCombos: number;
  eliminatedByVmem: number;
  fitCombos: number;
}

const CANDIDATE_POWERS = [16, 32, 64, 128, 256, 512];

function vmemMatmul(
  bm: number,
  bn: number,
  bk: number,
  spec: LayerSpec
): number {
  const ib = DTYPE_ITEMSIZE[spec.inputDtype];
  const ob = DTYPE_ITEMSIZE[spec.outputDtype];
  const ab = DTYPE_ITEMSIZE[spec.accumulatorDtype];
  return bm * bk * ib + bk * bn * ib + bm * bn * ob + bm * bn * ab;
}

function vmemRmsnorm(bm: number, bn: number, spec: LayerSpec): number {
  const ib = DTYPE_ITEMSIZE[spec.inputDtype];
  const ob = DTYPE_ITEMSIZE[spec.outputDtype];
  const ab = DTYPE_ITEMSIZE[spec.accumulatorDtype];
  // input tile + weight vector + output tile + per-row accumulator
  return bm * bn * ib + bn * ib + bm * bn * ob + bm * ab;
}

const aligned = (v: number, a: number) => v % a === 0;

/** Aligned candidate powers ≤ dim, plus the full dim (always valid). */
function candidates(dim: number, mustAlignTo: number): number[] {
  const result = CANDIDATE_POWERS.filter((p) => p <= dim && aligned(p, mustAlignTo));
  if (!result.includes(dim)) result.push(dim);
  return result.sort((a, b) => a - b);
}

function buildAxis(
  axis: Axis,
  dim: number,
  alignment: number,
  alignmentLabel: string
): AxisBreakdown {
  const kept = new Set(candidates(dim, alignment));
  const chips: CandidateChip[] = CANDIDATE_POWERS.map((value) => ({
    value,
    aligned: aligned(value, alignment),
    withinDim: value <= dim,
    isFullDim: false,
    kept: kept.has(value),
  }));
  // surface the full-dim exception as its own chip when it isn't a power already
  if (!CANDIDATE_POWERS.includes(dim)) {
    chips.push({
      value: dim,
      aligned: aligned(dim, alignment),
      withinDim: true,
      isFullDim: true,
      kept: true,
    });
  }
  chips.sort((a, b) => a.value - b.value);
  return { axis, dim, alignment, alignmentLabel, chips };
}

export function solve(spec: LayerSpec, hw: HardwareLimits): SolveResult {
  const budget = vmemBudgetBytes(hw);

  if (spec.opType === "matmul") {
    return solveMatmul(spec, hw, budget);
  }
  if (spec.opType === "rmsnorm") {
    return solveRmsnorm(spec, hw, budget);
  }
  return {
    ok: false,
    error: `Unsupported op_type: ${spec.opType}`,
    axes: [],
    budgetBytes: budget,
    vmemBytes: hw.vmemBytes,
    totalCombos: 0,
    eliminatedByVmem: 0,
    fitCombos: 0,
  };
}

function solveMatmul(
  spec: LayerSpec,
  hw: HardwareLimits,
  budget: number
): SolveResult {
  const mCands = candidates(spec.M, hw.sublaneWidth);
  const nCands = candidates(spec.N, hw.vectorWidth);
  const kCands = candidates(spec.K, hw.vectorWidth);

  const axes = [
    buildAxis("m", spec.M, hw.sublaneWidth, `% ${hw.sublaneWidth}`),
    buildAxis("n", spec.N, hw.vectorWidth, `% ${hw.vectorWidth}`),
    buildAxis("k", spec.K, hw.vectorWidth, `% ${hw.vectorWidth}`),
  ];

  let best: KernelConfig | null = null;
  let bestScore = -1;
  let eliminated = 0;
  let fit = 0;
  const total = mCands.length * nCands.length * kCands.length;

  // iterate largest → smallest, keep any fitting combo, maximize bm*bn*bk
  for (const bm of [...mCands].reverse()) {
    for (const bn of [...nCands].reverse()) {
      for (const bk of [...kCands].reverse()) {
        const vmem = vmemMatmul(bm, bn, bk, spec);
        if (vmem > budget) {
          eliminated++;
          continue;
        }
        fit++;
        const score = bm * bn * bk;
        if (score > bestScore) {
          bestScore = score;
          const numKTiles = Math.max(1, Math.floor(spec.K / bk));
          best = {
            blockM: bm,
            blockN: bn,
            blockK: bk,
            stages: 1,
            inputDtype: spec.inputDtype,
            outputDtype: spec.outputDtype,
            accumulatorDtype: spec.accumulatorDtype,
            totalVmemEstimateBytes: vmem,
            vmemUtilizationFraction: vmem / hw.vmemBytes,
            numKTiles,
          };
        }
      }
    }
  }

  return {
    ok: best !== null,
    error: best === null ? `No valid tile fits within the ${budget} B VMEM budget` : undefined,
    config: best ?? undefined,
    axes,
    budgetBytes: budget,
    vmemBytes: hw.vmemBytes,
    totalCombos: total,
    eliminatedByVmem: eliminated,
    fitCombos: fit,
  };
}

function solveRmsnorm(
  spec: LayerSpec,
  hw: HardwareLimits,
  budget: number
): SolveResult {
  const mCands = candidates(spec.M, hw.sublaneWidth);
  const nCands = candidates(spec.N, hw.vectorWidth);

  const axes = [
    buildAxis("m", spec.M, hw.sublaneWidth, `% ${hw.sublaneWidth}`),
    buildAxis("n", spec.N, hw.vectorWidth, `% ${hw.vectorWidth}`),
  ];

  let best: KernelConfig | null = null;
  let bestScore = -1;
  let eliminated = 0;
  let fit = 0;
  const total = mCands.length * nCands.length;

  for (const bm of [...mCands].reverse()) {
    for (const bn of [...nCands].reverse()) {
      const vmem = vmemRmsnorm(bm, bn, spec);
      if (vmem > budget) {
        eliminated++;
        continue;
      }
      fit++;
      const score = bm * bn;
      if (score > bestScore) {
        bestScore = score;
        best = {
          blockM: bm,
          blockN: bn,
          blockK: spec.K,
          stages: 1,
          inputDtype: spec.inputDtype,
          outputDtype: spec.outputDtype,
          accumulatorDtype: spec.accumulatorDtype,
          totalVmemEstimateBytes: vmem,
          vmemUtilizationFraction: vmem / hw.vmemBytes,
          numKTiles: 1,
        };
      }
    }
  }

  return {
    ok: best !== null,
    error: best === null ? `No valid tile fits within the ${budget} B VMEM budget` : undefined,
    config: best ?? undefined,
    axes,
    budgetBytes: budget,
    vmemBytes: hw.vmemBytes,
    totalCombos: total,
    eliminatedByVmem: eliminated,
    fitCombos: fit,
  };
}
