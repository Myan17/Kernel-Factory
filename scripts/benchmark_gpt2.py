"""
End-to-end benchmark: JAX XLA vs custom Pallas kernels on TPU v5e.
Model: GPT-2 small (117M) layer shapes + scaled-up variants.

Story:
  - Small shapes (512×N×K): XLA is tightly tuned → ~1x (baseline confirmation)
  - Larger shapes (2048+): custom tiling shows separation
  - Fused MatMul+RMSNorm: avoids 2 HBM round-trips → real speedup

Run:  uv run python scripts/benchmark_gpt2.py
"""
from __future__ import annotations

import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, "/home/abhis/kernel-factory/src")
from kernel_factory.schemas import DType, HardwareLimits, KernelConfig, LayerSpec  # noqa
from kernel_factory.solver import TileSolver  # noqa

# ── Config ────────────────────────────────────────────────────────────────────
HW       = HardwareLimits.for_v5e()
N_WARMUP = 5
N_RUNS   = 30
EPS      = 1e-6

# ── Layer shapes ──────────────────────────────────────────────────────────────
# GPT-2 small (117M): hidden=768, ffn=3072, heads=12, seq_len=512
# Scaled: seq_len=2048 (modern context window)
MATMUL_CASES = {
    # label                          M     N     K
    "gpt2 attn_qkv (512×768×768)":  (512,  768,  768),
    "gpt2 ffn_up   (512×768×3072)": (512,  3072, 768),
    "gpt2 ffn_down (512×3072×768)": (512,  768,  3072),
    "med  attn     (2048×768×768)": (2048, 768,  768),
    "med  ffn_up   (2048×768×3072)":(2048, 3072, 768),
    "large matmul  (2048×2048×2048)":(2048,2048, 2048),
}
RMSNORM_CASES = {
    "gpt2 rmsnorm  (512×768)":     (512,  768),
    "med  rmsnorm  (2048×768)":    (2048, 768),
}

# ── Block size: ensures K % block_k == 0 (fixes solver edge-case) ─────────────
_CANDS = [16, 32, 64, 128, 256, 512]

def _largest_div(dim: int, cands: list[int], align: int) -> int:
    valid = [p for p in reversed(cands)
             if p <= dim and dim % p == 0 and p % align == 0]
    return valid[0] if valid else align

def _matmul_config(M, N, K) -> KernelConfig:
    bm = _largest_div(M, _CANDS, HW.sublane_width)
    bn = _largest_div(N, _CANDS, HW.vector_width)
    bk = _largest_div(K, _CANDS, HW.vector_width)
    spec = LayerSpec(op_type="matmul", M=M, N=N, K=K)
    ib, ob, ab = spec.input_dtype.itemsize, spec.output_dtype.itemsize, spec.accumulator_dtype.itemsize
    vmem = bm*bk*ib + bk*bn*ib + bm*bn*ob + bm*bn*ab
    return KernelConfig(
        block_m=bm, block_n=bn, block_k=bk, stages=1,
        input_dtype=spec.input_dtype, output_dtype=spec.output_dtype,
        accumulator_dtype=spec.accumulator_dtype,
        total_vmem_estimate_bytes=vmem,
        vmem_utilization_fraction=vmem / HW.vmem_bytes,
    )

def _rmsnorm_config(M, N) -> KernelConfig:
    # Full-row tile: bm rows × full N columns per Pallas tile.
    # Correct RMSNorm requires the entire row to compute mean(x²).
    # Float32 output avoids bfloat16 (8,128) tiling restriction on N=768.
    bm = _largest_div(M, _CANDS, HW.sublane_width)
    vmem = bm * N * 4 + N * 2 + bm * N * 4   # x(f32) + w(bf16) + o(f32)
    spec = LayerSpec(op_type="rmsnorm", M=M, N=N, K=N)
    return KernelConfig(
        block_m=bm, block_n=N, block_k=N, stages=1,
        input_dtype=spec.input_dtype, output_dtype=spec.output_dtype,
        accumulator_dtype=spec.accumulator_dtype,
        total_vmem_estimate_bytes=vmem,
        vmem_utilization_fraction=vmem / HW.vmem_bytes,
    )

# ── Kernel builders ───────────────────────────────────────────────────────────

def make_pallas_matmul(M, N, K, cfg: KernelConfig):
    """Native Pallas tiled matmul — no interpret=True."""
    bm, bn, bk = cfg.block_m, cfg.block_n, cfg.block_k
    num_k = K // bk

    def kernel(a_ref, b_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)
        acc_ref[...] += jnp.dot(
            a_ref[...].astype(jnp.float32),
            b_ref[...].astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        @pl.when(pl.program_id(2) == num_k - 1)
        def _():
            o_ref[...] = acc_ref[...].astype(jnp.bfloat16)

    return jax.jit(pl.pallas_call(
        kernel,
        jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(M // bm, N // bn, num_k),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
            pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    ))


def make_pallas_rmsnorm(M, N, cfg: KernelConfig):
    """
    Correct RMSNorm: full-row tile so mean(x²) is computed over all N elements.
    Key: w is passed as float32 2D (1,N) to avoid Mosaic 1D-bf16 tiling issue.
    Output float32 (cast to bf16 outside) to avoid N=768 output tiling issue.
    """
    bm = cfg.block_m

    def kernel(x_ref, w_ref, o_ref):
        x_f  = x_ref[...].astype(jnp.float32)      # (bm, N)
        w_f  = w_ref[0, :]                          # (N,) — w is stored as (1,N)
        rms  = jnp.sqrt(jnp.mean(x_f ** 2, axis=-1, keepdims=True) + EPS)
        o_ref[...] = (x_f / rms) * w_f              # (bm, N) float32

    pallas_fn = pl.pallas_call(
        kernel,
        jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // bm,),
        in_specs=[
            pl.BlockSpec((bm, N), lambda m: (m, 0)),  # x: (bm, N) bf16
            pl.BlockSpec((1, N),  lambda m: (0, 0)),  # w: (1, N) f32 — no 1D tiling
        ],
        out_specs=pl.BlockSpec((bm, N), lambda m: (m, 0)),
    )
    # Reshape w: (N,) → (1,N) as f32, cast output back to bf16
    return jax.jit(
        lambda x, w: pallas_fn(x, w.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)
    )


def make_fused_matmul_norm(M, N, K, cfg: KernelConfig):
    """
    Fused MatMul + RMSNorm in ONE Pallas kernel.
    - Accumulates K tiles of A×B into VMEM (bm, N) — stays in VMEM the whole time.
    - On last K tile: applies RMSNorm in-place before writing output.
    - Saves: 1 full HBM write of (M, N) matrix + 1 full HBM read for norm.
    - Output: float32 to avoid bfloat16 N=768 tiling restriction.
    """
    bm, bk = cfg.block_m, cfg.block_k
    num_k  = K // bk

    # VMEM: accumulator(bm,N)×f32 + a-tile(bm,bk)×bf16 + b-tile(bk,N)×bf16
    vmem_needed = bm*N*4 + bm*bk*2 + bk*N*2
    if vmem_needed > HW.vmem_budget_bytes:
        return None, f"VMEM {vmem_needed//1024//1024}MiB > budget"

    def kernel(a_ref, b_ref, w_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(1) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)   # (bm, N) f32 in VMEM

        acc_ref[...] += jnp.dot(
            a_ref[...].astype(jnp.float32),
            b_ref[...].astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )

        @pl.when(pl.program_id(1) == num_k - 1)
        def _():
            x   = acc_ref[...]                               # (bm, N) f32
            w_f = w_ref[0, :]                                # (N,) from (1, N)
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS)
            o_ref[...] = (x / rms) * w_f                    # f32 out

    fn = pl.pallas_call(
        kernel,
        jax.ShapeDtypeStruct((M, N), jnp.float32),    # float32 — cast outside
        grid=(M // bm, num_k),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda m, k: (m, k)),   # a tile
            pl.BlockSpec((bk, N),  lambda m, k: (k, 0)),   # full N cols of b
            pl.BlockSpec((1, N),   lambda m, k: (0, 0)),   # w as (1,N) f32
        ],
        out_specs=pl.BlockSpec((bm, N), lambda m, k: (m, 0)),
        scratch_shapes=[pltpu.VMEM((bm, N), jnp.float32)],  # acc stays in VMEM
    )
    return jax.jit(
        lambda a, b, w: fn(a, b, w.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)
    ), None


# ── Benchmark harness ─────────────────────────────────────────────────────────

def tflops(M, N, K, t_s):
    return 2 * M * N * K / t_s / 1e12

def bench(fn, *args) -> float:
    for _ in range(N_WARMUP):
        fn(*args).block_until_ready()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times)


# ── Main ──────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'─'*90}")
    print(f"  {title}")
    print('─'*90)
    hdr = (f"{'Op':<35} {'Tile m/n/k':<16} {'XLA ms':>8} {'Pallas ms':>10} "
           f"{'XLA TFLOPS':>11} {'Pallas TFLOPS':>14} {'Speedup':>9}")
    print(hdr)
    print('·'*len(hdr))

def row(label, tile, xla_t, pallas_t, M, N, K):
    xla_ms  = xla_t * 1e3
    xla_tf  = tflops(M, N, K, xla_t)
    if pallas_t is not None:
        p_ms    = pallas_t * 1e3
        p_tf    = tflops(M, N, K, pallas_t)
        speedup = xla_t / pallas_t
        return (f"{label:<35} {tile:<16} {xla_ms:>8.3f} {p_ms:>10.3f} "
                f"{xla_tf:>11.3f} {p_tf:>14.3f} {speedup:>8.2f}x")
    else:
        return (f"{label:<35} {tile:<16} {xla_ms:>8.3f} {'ERR':>10} "
                f"{xla_tf:>11.3f} {'—':>14} {'—':>9}")

def main():
    print(f"\n{'='*90}")
    print(f"  TPU Kernel Factory — GPT-2 small full-pipeline benchmark")
    print(f"  JAX {jax.__version__}  |  Device: {jax.devices()[0]}")
    print(f"  Goal: custom Pallas vs JAX XLA — latency (ms) and TFLOPS")
    print(f"{'='*90}")

    key = jax.random.PRNGKey(42)
    all_results = []

    # ── Section 1: MatMul ────────────────────────────────────────────────────
    section("MatMul:  JAX XLA (jit+jnp.matmul) vs custom Pallas tiled kernel")
    for label, (M, N, K) in MATMUL_CASES.items():
        cfg  = _matmul_config(M, N, K)
        a    = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
        b    = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
        xla_t = bench(jax.jit(jnp.matmul), a, b)
        try:
            fn = make_pallas_matmul(M, N, K, cfg)
            p_t = bench(fn, a, b)
        except Exception:
            p_t = None
        tile = f"{cfg.block_m}/{cfg.block_n}/{cfg.block_k}"
        print(row(label, tile, xla_t, p_t, M, N, K))
        all_results.append((label, xla_t, p_t))

    # ── Section 2: RMSNorm ───────────────────────────────────────────────────
    section("RMSNorm: JAX XLA (jit+manual) vs Pallas tile-wise norm kernel")
    for label, (M, N) in RMSNORM_CASES.items():
        cfg  = _rmsnorm_config(M, N)
        x    = jax.random.normal(key, (M, N), dtype=jnp.bfloat16)
        w    = jax.random.normal(key, (N,),   dtype=jnp.bfloat16)

        def jax_rms(x, w):
            xf  = x.astype(jnp.float32)
            rms = jnp.sqrt(jnp.mean(xf**2, axis=-1, keepdims=True) + EPS)
            return ((xf / rms) * w.astype(jnp.float32)).astype(jnp.bfloat16)

        xla_t = bench(jax.jit(jax_rms), x, w)
        try:
            fn = make_pallas_rmsnorm(M, N, cfg)
            p_t = bench(fn, x, w)
        except Exception:
            traceback.print_exc()
            p_t = None
        tile = f"{cfg.block_m}/{cfg.block_n}/—"
        flops = 5 * M * N
        def rms_row(label, tile, xla_t, p_t, flops):
            xla_ms = xla_t * 1e3
            xla_tf = flops / xla_t / 1e12
            if p_t:
                return (f"{label:<35} {tile:<16} {xla_ms:>8.3f} {p_t*1e3:>10.3f} "
                        f"{xla_tf:>11.6f} {flops/p_t/1e12:>14.6f} {xla_t/p_t:>8.2f}x")
            return (f"{label:<35} {tile:<16} {xla_ms:>8.3f} {'ERR':>10} "
                    f"{xla_tf:>11.6f} {'—':>14} {'—':>9}")
        print(rms_row(label, tile, xla_t, p_t, flops))
        all_results.append((label, xla_t, p_t))

    # ── Section 3: Fused MatMul + RMSNorm ────────────────────────────────────
    section("FUSED MatMul+RMSNorm: XLA (2 ops) vs single Pallas kernel (saves 1 HBM round-trip)")
    fused_cases = {
        "gpt2 fused (512×768×768)":   (512,  768,  768),
        "med  fused (2048×768×768)":   (2048, 768,  768),
    }
    for label, (M, N, K) in fused_cases.items():
        cfg = _matmul_config(M, N, K)
        a   = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
        b   = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
        w   = jax.random.normal(key, (N,),   dtype=jnp.bfloat16)

        def xla_fused(a, b, w):
            c  = jnp.matmul(a, b)
            cf = c.astype(jnp.float32)
            rms = jnp.sqrt(jnp.mean(cf**2, axis=-1, keepdims=True) + EPS)
            return ((cf / rms) * w.astype(jnp.float32)).astype(jnp.bfloat16)

        xla_t = bench(jax.jit(xla_fused), a, b, w)

        fn, err = make_fused_matmul_norm(M, N, K, cfg)
        if fn is not None:
            try:
                p_t = bench(fn, a, b, w)
            except Exception:
                traceback.print_exc()
                p_t = None
                err = "compile/run failed"
        else:
            p_t = None

        tile = f"{cfg.block_m}/{N}/{cfg.block_k}"
        note = f"  [{err}]" if err and p_t is None else ""
        print(row(label, tile, xla_t, p_t, M, N, K) + note)
        all_results.append((label, xla_t, p_t))

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [(l, x, p) for l, x, p in all_results if p is not None and p > 0]
    print(f"\n{'='*90}")
    if valid:
        avg_speedup = sum(x/p for _, x, p in valid) / len(valid)
        best = max(valid, key=lambda r: r[1]/r[2])
        print(f"  Ran {len(all_results)} kernels  |  {len(valid)} succeeded")
        print(f"  Average speedup (Pallas vs XLA):  {avg_speedup:.2f}x")
        print(f"  Best result: {best[0].strip()}  →  {best[1]/best[2]:.2f}x faster")
    else:
        print("  No successful Pallas runs to summarize.")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
