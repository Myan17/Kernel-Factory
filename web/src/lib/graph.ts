// Knowledge Graph schema (Kuzu) — 11 node types, 10 edge types. Tracks the
// provenance of every spec, tile, kernel, compile, benchmark, failure and fix.

export type NodeCategory = "spec" | "hardware" | "kernel" | "result" | "knowledge";

export interface KGNode {
  id: string;
  label: string;
  category: NodeCategory;
  blurb: string;
}

export interface KGEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export const KG_NODES: KGNode[] = [
  { id: "KernelTemplate", label: "KernelTemplate", category: "knowledge", blurb: "Verified Pallas skeleton with {placeholders} only." },
  { id: "KernelSpec", label: "KernelSpec", category: "spec", blurb: "The op + (M, N, K) + dtypes to generate." },
  { id: "HardwareLimits", label: "HardwareLimits", category: "hardware", blurb: "VMEM, bandwidth, vector/sublane widths per TPU." },
  { id: "TileSpec", label: "TileSpec", category: "spec", blurb: "Solver-computed (block_m, block_n, block_k)." },
  { id: "GeneratedKernel", label: "GeneratedKernel", category: "kernel", blurb: "Assembled, runnable JAX/Pallas source." },
  { id: "TestCase", label: "TestCase", category: "result", blurb: "Numeric check vs a pure-JAX baseline." },
  { id: "CompileResult", label: "CompileResult", category: "result", blurb: "Did it compile in CPU interpret-mode?" },
  { id: "BenchmarkResult", label: "BenchmarkResult", category: "result", blurb: "Latency / TFLOPS vs the XLA baseline." },
  { id: "FailureCase", label: "FailureCase", category: "knowledge", blurb: "A captured failing generation + its trace." },
  { id: "KnownBug", label: "KnownBug", category: "knowledge", blurb: "A catalogued failure mode (e.g. Mosaic version)." },
  { id: "FixPattern", label: "FixPattern", category: "knowledge", blurb: "A reusable remedy for a known bug." },
];

export const KG_EDGES: KGEdge[] = [
  { id: "e1", source: "KernelSpec", target: "KernelTemplate", label: "USES_TEMPLATE" },
  { id: "e2", source: "KernelSpec", target: "HardwareLimits", label: "CONSTRAINED_BY" },
  { id: "e3", source: "KernelSpec", target: "TileSpec", label: "HAS_TILE" },
  { id: "e4", source: "GeneratedKernel", target: "TileSpec", label: "GENERATED_FROM" },
  { id: "e5", source: "GeneratedKernel", target: "CompileResult", label: "HAS_COMPILE_RESULT" },
  { id: "e6", source: "GeneratedKernel", target: "BenchmarkResult", label: "HAS_BENCHMARK" },
  { id: "e7", source: "TestCase", target: "GeneratedKernel", label: "VALIDATES" },
  { id: "e8", source: "GeneratedKernel", target: "FailureCase", label: "CAUSED_FAILURE" },
  { id: "e9", source: "KnownBug", target: "FailureCase", label: "KNOWN_BUG_FOR" },
  { id: "e10", source: "FailureCase", target: "FixPattern", label: "FIXED_BY" },
];

export const CATEGORY_COLOR: Record<NodeCategory, string> = {
  spec: "var(--indigo)",
  hardware: "var(--duck)",
  kernel: "var(--violet)",
  result: "var(--mint)",
  knowledge: "var(--coral)",
};

export const CATEGORY_LABEL: Record<NodeCategory, string> = {
  spec: "Spec",
  hardware: "Hardware",
  kernel: "Kernel",
  result: "Result",
  knowledge: "Knowledge",
};
