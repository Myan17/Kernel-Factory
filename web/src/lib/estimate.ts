// Plausible, clearly-labelled estimates for the Verify stage readout. These are
// NOT measured — they are back-of-envelope numbers from peak roofline so the
// latency responds sensibly to the inputs. Real measured numbers live in the
// Proof section's benchmark dataset.

import type { HardwareLimits } from "./hardware";
import { DTYPE_ITEMSIZE } from "./hardware";
import type { KernelConfig, LayerSpec } from "./solver";

// bf16 peak TFLOP/s (rough public figures) used only for the estimate.
const PEAK_TFLOPS: Record<string, number> = { v4: 275, v5e: 197, v6e: 918 };
const EFFICIENCY = 0.32; // assume ~32% of peak for a single fused tile

export function estimateLatencyMs(
  spec: LayerSpec,
  hw: HardwareLimits,
  _config: KernelConfig
): number {
  if (spec.opType === "matmul") {
    const flops = 2 * spec.M * spec.N * spec.K;
    const peak = (PEAK_TFLOPS[hw.tpuVersion] ?? 197) * 1e12 * EFFICIENCY;
    return (flops / peak) * 1000;
  }
  // rmsnorm is bandwidth-bound
  const ib = DTYPE_ITEMSIZE[spec.inputDtype];
  const ob = DTYPE_ITEMSIZE[spec.outputDtype];
  const bytes = spec.M * spec.N * ib + spec.N * ib + spec.M * spec.N * ob;
  const bw = hw.hbmBandwidthGbps * 1e9 * 0.6;
  return (bytes / bw) * 1000;
}

/** Deterministic, plausible max-abs-error vs the pure-JAX baseline (< 1e-2). */
export function estimateMaxAbsError(spec: LayerSpec, config: KernelConfig): number {
  if (config.outputDtype === "float32") return 7.6e-6;
  // bf16 has ~8 mantissa bits; accumulation over K grows the error mildly.
  const k = spec.opType === "matmul" ? spec.K : spec.N;
  const base = 3.1e-3;
  const grow = Math.min(2.4e-3, Math.log2(Math.max(2, k)) * 1.9e-4);
  return Number((base + grow).toFixed(6));
}
