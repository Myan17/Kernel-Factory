// Hardware limits — ported 1:1 from kernel_factory/schemas.py (HardwareLimits).
// These numbers are the real constraints the Python solver runs against.

export type DType = "float32" | "bfloat16" | "int8";

export const DTYPE_ITEMSIZE: Record<DType, number> = {
  float32: 4,
  bfloat16: 2,
  int8: 1,
};

export type TpuVersion = "v4" | "v5e" | "v6e";

export interface HardwareLimits {
  tpuVersion: TpuVersion;
  vmemBytes: number;
  hbmBandwidthGbps: number;
  vectorWidth: number; // last dim must be a multiple of this
  sublaneWidth: number; // second-to-last dim must be a multiple of this
  vmemSafetyFraction: number;
}

const MiB = 1024 * 1024;

export const HARDWARE: Record<TpuVersion, HardwareLimits> = {
  v4: {
    tpuVersion: "v4",
    vmemBytes: 16 * MiB,
    hbmBandwidthGbps: 614.4,
    vectorWidth: 128,
    sublaneWidth: 8,
    vmemSafetyFraction: 0.75,
  },
  v5e: {
    tpuVersion: "v5e",
    vmemBytes: 128 * MiB,
    hbmBandwidthGbps: 819.2,
    vectorWidth: 128,
    sublaneWidth: 8,
    vmemSafetyFraction: 0.75,
  },
  v6e: {
    tpuVersion: "v6e",
    vmemBytes: 128 * MiB,
    hbmBandwidthGbps: 1638.4,
    vectorWidth: 128,
    sublaneWidth: 8,
    vmemSafetyFraction: 0.75,
  },
};

/** Usable VMEM after the safety margin (default 75% of total). */
export function vmemBudgetBytes(hw: HardwareLimits): number {
  return Math.floor(hw.vmemBytes * hw.vmemSafetyFraction);
}

export const TPU_LABELS: Record<TpuVersion, string> = {
  v4: "TPU v4",
  v5e: "TPU v5e",
  v6e: "TPU v6e",
};

export function formatBytes(bytes: number): string {
  if (bytes >= MiB) return `${(bytes / MiB).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}
