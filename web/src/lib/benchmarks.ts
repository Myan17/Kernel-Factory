// Real benchmark data — TPU v5e (v5litepod-1, us-south1-a), GPT-2 small, bf16,
// JAX 0.6.2. Latency = min over 30 runs. speedup = XLA_ms / Pallas_ms
// (>1.0 means the generated Pallas kernel wins). These are measured numbers;
// the honest headline is 0.95× average with targeted wins, NOT a blanket speedup.

export interface Benchmark {
  op: string;
  shape: string;
  M: number;
  N: number;
  K: number | null;
  xlaMs: number;
  pallasMs: number;
  xlaTflops: number;
  pallasTflops: number;
  speedup: number;
  win: boolean;
  kind: "matmul" | "rmsnorm" | "fused";
}

export const BENCHMARKS: Benchmark[] = [
  { op: "attn_qkv", shape: "512×768×768", M: 512, N: 768, K: 768, xlaMs: 0.108, pallasMs: 0.121, xlaTflops: 5.604, pallasTflops: 5.0, speedup: 0.89, win: false, kind: "matmul" },
  { op: "ffn_up", shape: "512×768→3072", M: 512, N: 3072, K: 768, xlaMs: 0.13, pallasMs: 0.138, xlaTflops: 18.585, pallasTflops: 17.54, speedup: 0.94, win: false, kind: "matmul" },
  { op: "ffn_down", shape: "512×3072→768", M: 512, N: 768, K: 3072, xlaMs: 0.13, pallasMs: 0.124, xlaTflops: 18.581, pallasTflops: 19.559, speedup: 1.05, win: true, kind: "matmul" },
  { op: "med attn", shape: "2048×768×768", M: 2048, N: 768, K: 768, xlaMs: 0.117, pallasMs: 0.132, xlaTflops: 20.698, pallasTflops: 18.269, speedup: 0.88, win: false, kind: "matmul" },
  { op: "med ffn_up", shape: "2048×768→3072", M: 2048, N: 3072, K: 768, xlaMs: 0.158, pallasMs: 0.194, xlaTflops: 61.101, pallasTflops: 49.864, speedup: 0.82, win: false, kind: "matmul" },
  { op: "large mm", shape: "2048×2048×2048", M: 2048, N: 2048, K: 2048, xlaMs: 0.192, pallasMs: 0.225, xlaTflops: 89.302, pallasTflops: 76.24, speedup: 0.85, win: false, kind: "matmul" },
  { op: "rmsnorm", shape: "512×768", M: 512, N: 768, K: null, xlaMs: 0.104, pallasMs: 0.101, xlaTflops: 0.0189, pallasTflops: 0.0195, speedup: 1.03, win: true, kind: "rmsnorm" },
  { op: "med rmsnorm", shape: "2048×768", M: 2048, N: 768, K: null, xlaMs: 0.111, pallasMs: 0.107, xlaTflops: 0.0709, pallasTflops: 0.0734, speedup: 1.04, win: true, kind: "rmsnorm" },
  { op: "fused mm+norm", shape: "512×768×768", M: 512, N: 768, K: 768, xlaMs: 0.104, pallasMs: 0.105, xlaTflops: 5.799, pallasTflops: 5.777, speedup: 1.0, win: false, kind: "fused" },
  { op: "fused mm+norm", shape: "2048×768×768", M: 2048, N: 768, K: 768, xlaMs: 0.12, pallasMs: 0.121, xlaTflops: 20.118, pallasTflops: 19.928, speedup: 0.99, win: false, kind: "fused" },
];

export const HEADLINE = {
  kernelsRan: 10,
  kernelsTotal: 10,
  testsPass: 92,
  testsTotal: 92,
  avgSpeedup: 0.95,
  rmsnormGain: "+3–4%",
  ffnDownGain: "+5%",
  headroom: "+20–50%",
  device: "Google Cloud TPU v5e",
  devicePod: "v5litepod-1, us-south1-a",
  jaxVersion: "JAX 0.6.2",
};

export const avgSpeedup = (rows: Benchmark[]): number =>
  rows.reduce((s, r) => s + r.speedup, 0) / rows.length;
