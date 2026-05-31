import { describe, it, expect } from "vitest";
import { HARDWARE, vmemBudgetBytes } from "./hardware";
import { solve, type LayerSpec, type OpType } from "./solver";

// Ground truth captured by running the REAL kernel_factory/solver.py against
// each spec (see web/README.md). These assertions prove the TS port is faithful
// to the shipped Python — the live demo computes, it does not fake.

function spec(opType: OpType, M: number, N: number, K: number): LayerSpec {
  return {
    opType,
    M,
    N,
    K,
    inputDtype: "bfloat16",
    outputDtype: "bfloat16",
    accumulatorDtype: "float32",
  };
}

interface Expected {
  bm: number;
  bn: number;
  bk: number;
  vmem: number;
  util: number;
  nkt: number;
}

const CASES: Array<{
  name: string;
  tpu: "v4" | "v5e" | "v6e";
  op: OpType;
  M: number;
  N: number;
  K: number;
  expect: Expected;
}> = [
  // GPT-2 small attention QKV projection on v5e
  { name: "attn_qkv", tpu: "v5e", op: "matmul", M: 512, N: 768, K: 768,
    expect: { bm: 512, bn: 768, bk: 768, vmem: 4325376, util: 0.0322265625, nkt: 1 } },
  // FFN up-projection
  { name: "ffn_up", tpu: "v5e", op: "matmul", M: 512, N: 3072, K: 768,
    expect: { bm: 512, bn: 3072, bk: 768, vmem: 14942208, util: 0.111328125, nkt: 1 } },
  // FFN down-projection (asymmetric K-reduction)
  { name: "ffn_down", tpu: "v5e", op: "matmul", M: 512, N: 768, K: 3072,
    expect: { bm: 512, bn: 768, bk: 3072, vmem: 10223616, util: 0.076171875, nkt: 1 } },
  // Large square matmul
  { name: "large_mm", tpu: "v5e", op: "matmul", M: 2048, N: 2048, K: 2048,
    expect: { bm: 2048, bn: 2048, bk: 2048, vmem: 41943040, util: 0.3125, nkt: 1 } },
  // RMSNorm — block_n is the full row (full-dim exception), exceeds the 128-set
  { name: "rmsnorm", tpu: "v5e", op: "rmsnorm", M: 512, N: 768, K: 768,
    expect: { bm: 512, bn: 768, bk: 768, vmem: 1576448, util: 0.011745452880859375, nkt: 1 } },
  { name: "med_rmsnorm", tpu: "v5e", op: "rmsnorm", M: 2048, N: 768, K: 768,
    expect: { bm: 2048, bn: 768, bk: 768, vmem: 6301184, util: 0.046947479248046875, nkt: 1 } },
  // v4 has only 16 MiB VMEM (12 MiB budget): the solver MUST shrink a tile.
  // Here it lands the winner at exactly 75% utilization.
  { name: "v4_large", tpu: "v4", op: "matmul", M: 2048, N: 2048, K: 2048,
    expect: { bm: 2048, bn: 256, bk: 2048, vmem: 12582912, util: 0.75, nkt: 1 } },
  { name: "v4_4096", tpu: "v4", op: "matmul", M: 4096, N: 4096, K: 4096,
    expect: { bm: 512, bn: 512, bk: 4096, vmem: 9961472, util: 0.59375, nkt: 1 } },
  { name: "v4_tiny", tpu: "v4", op: "matmul", M: 64, N: 64, K: 64,
    expect: { bm: 64, bn: 64, bk: 64, vmem: 40960, util: 0.00244140625, nkt: 1 } },
  { name: "default_mm", tpu: "v5e", op: "matmul", M: 1024, N: 1024, K: 512,
    expect: { bm: 1024, bn: 1024, bk: 512, vmem: 8388608, util: 0.0625, nkt: 1 } },
];

describe("solver matches the real Python solver.py", () => {
  for (const c of CASES) {
    it(`${c.name} (${c.tpu} ${c.op} ${c.M}×${c.N}×${c.K})`, () => {
      const r = solve(spec(c.op, c.M, c.N, c.K), HARDWARE[c.tpu]);
      expect(r.ok).toBe(true);
      const cfg = r.config!;
      expect(cfg.blockM).toBe(c.expect.bm);
      expect(cfg.blockN).toBe(c.expect.bn);
      expect(cfg.blockK).toBe(c.expect.bk);
      expect(cfg.totalVmemEstimateBytes).toBe(c.expect.vmem);
      expect(cfg.vmemUtilizationFraction).toBeCloseTo(c.expect.util, 12);
      expect(cfg.numKTiles).toBe(c.expect.nkt);
      // never exceed the safety budget
      expect(cfg.totalVmemEstimateBytes).toBeLessThanOrEqual(
        vmemBudgetBytes(HARDWARE[c.tpu])
      );
    });
  }
});

describe("alignment rules", () => {
  it("block_n and block_k are multiples of vector_width (128)", () => {
    const r = solve(spec("matmul", 1024, 1024, 512), HARDWARE.v5e);
    expect(r.config!.blockN % 128).toBe(0);
    expect(r.config!.blockK % 128).toBe(0);
  });

  it("block_m is a multiple of sublane_width (8)", () => {
    const r = solve(spec("matmul", 1024, 1024, 512), HARDWARE.v5e);
    expect(r.config!.blockM % 8).toBe(0);
  });

  it("crosses out 16/32/64 from the N axis (fail % 128)", () => {
    const r = solve(spec("matmul", 1024, 1024, 512), HARDWARE.v5e);
    const nAxis = r.axes.find((a) => a.axis === "n")!;
    for (const v of [16, 32, 64]) {
      const chip = nAxis.chips.find((c) => c.value === v)!;
      expect(chip.aligned).toBe(false);
      expect(chip.kept).toBe(false);
    }
    for (const v of [128, 256, 512]) {
      const chip = nAxis.chips.find((c) => c.value === v)!;
      expect(chip.kept).toBe(true);
    }
  });
});

describe("VMEM elimination accounting", () => {
  it("v4 large matmul eliminates combos by VMEM budget", () => {
    const r = solve(spec("matmul", 2048, 2048, 2048), HARDWARE.v4);
    expect(r.eliminatedByVmem).toBeGreaterThan(0);
    expect(r.config!.vmemUtilizationFraction).toBeLessThanOrEqual(0.75);
  });

  it("v5e GPT-2 shapes fit with headroom (no VMEM elimination)", () => {
    const r = solve(spec("matmul", 512, 768, 768), HARDWARE.v5e);
    expect(r.eliminatedByVmem).toBe(0);
  });
});

describe("error states", () => {
  it("flags an unsupported op cleanly instead of crashing", () => {
    const r = solve(
      { ...spec("matmul", 512, 512, 512), opType: "attention" as OpType },
      HARDWARE.v5e
    );
    expect(r.ok).toBe(false);
    expect(r.error).toMatch(/Unsupported op_type/);
  });
});
