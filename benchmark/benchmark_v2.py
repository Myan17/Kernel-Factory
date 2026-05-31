"""
benchmark_v2.py — Winnable TPU benchmark.

Three sections:
  A. Standalone MatMul — expect parity with XLA (≥0.98×) after pipeline fix
     (dimension_semantics + double-buffered VMEM solver)
  B. Fused MatMul+RMSNorm at large batch — expect ≥1.15× at batch≥8192
     (intermediate tensor never written to HBM)
  C. FlashAttention vs jax.nn.dot_product_attention — expect ≥1.5× at seq≥1024
     (score matrix stays in VMEM, never touches HBM)

Run on a TPU v5e (v5litepod-1 or larger):
    uv run python benchmark/benchmark_v2.py

On CPU (no TPU) the script still runs via JAX's default backend; timings will
reflect CPU performance and not the TPU targets above.
"""
from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from kernel_factory.assembler import Assembler
from kernel_factory.schemas import HardwareLimits, LayerSpec
from kernel_factory.solver import TileSolver

console = Console()
HW      = HardwareLimits.for_v5e()
WARMUP  = 5
ITERS   = 50


# ── Timing helper ─────────────────────────────────────────────────────────────

def timed_ms(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """JIT-compile + warm up, then time `iters` runs. Returns median ms."""
    jit_fn = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(jit_fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(*args))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _make_pallas_fn(spec: LayerSpec):
    """Assemble and exec() a Pallas kernel, return its run_* function."""
    config   = TileSolver(HW).solve(spec)
    code     = Assembler().assemble(spec, config)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    fn_name = {
        "matmul":               "run_matmul",
        "fused_matmul_rmsnorm": "run_fused_matmul_rmsnorm",
        "flash_attention":      "run_flash_attention",
    }[spec.op_type]
    return ns[fn_name]


# ── Section A — Standalone MatMul ─────────────────────────────────────────────

MATMUL_SHAPES = [
    ("gpt2 attn_qkv  (512×768×768)",    512,  768,  768),
    ("gpt2 ffn_up    (512×768→3072)",   512, 3072,  768),
    ("med  attn      (2048×768×768)",  2048,  768,  768),
    ("med  ffn_up    (2048×768→3072)", 2048, 3072,  768),
    ("large matmul   (2048×2048×2048)",2048, 2048, 2048),
]


def run_section_a() -> None:
    table = Table(
        title="Section A — Standalone MatMul: XLA vs Pallas (after pipeline fix)\n"
              "Target: speedup ≥ 0.98× for all shapes"
    )
    for col in ["Shape", "XLA ms", "Pallas ms", "Speedup"]:
        table.add_column(col, justify="right" if col != "Shape" else "left")

    key = jax.random.PRNGKey(0)
    for label, M, N, K in MATMUL_SHAPES:
        a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)

        xla_fn = lambda a, b: jax.lax.dot_general(  # noqa: E731
            a, b, (([1], [0]), ([], []))
        ).astype(jnp.bfloat16)

        spec   = LayerSpec(op_type="matmul", M=M, N=N, K=K)
        pal_fn = _make_pallas_fn(spec)

        xla_ms  = timed_ms(xla_fn, a, b)
        pal_ms  = timed_ms(pal_fn, a, b)
        speedup = xla_ms / pal_ms

        colour = "green" if speedup >= 0.98 else "red"
        table.add_row(
            label,
            f"{xla_ms:.3f}",
            f"{pal_ms:.3f}",
            f"[{colour}]{speedup:.2f}×[/{colour}]",
        )

    console.print(table)


# ── Section B — Fused MatMul+RMSNorm ─────────────────────────────────────────

FUSION_SHAPES = [
    ("batch=512   (512×768×768)",   512,  768, 768),
    ("batch=2048  (2048×768×768)", 2048,  768, 768),
    ("batch=4096  (4096×768×768)", 4096,  768, 768),
    ("batch=8192  (8192×768×768)", 8192,  768, 768),
]


def run_section_b() -> None:
    table = Table(
        title="Section B — Fused MatMul+RMSNorm: XLA 2-op vs Pallas fused\n"
              "Win zone: batch≥4096 (intermediate tensor > on-chip capacity)"
    )
    for col in ["Shape", "XLA 2-op ms", "Pallas fused ms", "Speedup", "HBM saved"]:
        table.add_column(col, justify="right" if col != "Shape" else "left")

    key = jax.random.PRNGKey(1)
    for label, M, N, K in FUSION_SHAPES:
        a = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
        w = jax.random.normal(key, (N,),   dtype=jnp.bfloat16)

        def xla_2op(a, b, w):
            x   = jax.lax.dot_general(a, b, (([1], [0]), ([], []))).astype(jnp.float32)
            rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            return ((x / rms) * w.astype(jnp.float32)).astype(jnp.bfloat16)

        spec   = LayerSpec(op_type="fused_matmul_rmsnorm", M=M, N=N, K=K)
        pal_fn = _make_pallas_fn(spec)

        xla_ms  = timed_ms(xla_2op, a, b, w)
        pal_ms  = timed_ms(pal_fn,  a, b, w)
        speedup = xla_ms / pal_ms

        saved_mb = M * N * 2 / (1024 ** 2)
        colour   = "green"  if speedup >= 1.05 else (
                   "yellow" if speedup >= 0.98 else "red")
        table.add_row(
            label,
            f"{xla_ms:.3f}",
            f"{pal_ms:.3f}",
            f"[{colour}]{speedup:.2f}×[/{colour}]",
            f"{saved_mb:.1f} MB",
        )

    console.print(table)


# ── Section C — FlashAttention ────────────────────────────────────────────────

ATTN_SHAPES = [
    # (label,            batch, heads, seq,  head_dim)
    ("seq=512  h=12 d=64",  1, 12,  512, 64),
    ("seq=1024 h=12 d=64",  1, 12, 1024, 64),
    ("seq=2048 h=12 d=64",  1, 12, 2048, 64),
    ("seq=4096 h=12 d=64",  1, 12, 4096, 64),
]


def run_section_c() -> None:
    table = Table(
        title="Section C — FlashAttention: XLA (materialise S) vs Pallas (S stays in VMEM)\n"
              "Win zone: seq≥1024 — score matrix too large for HBM-free XLA path"
    )
    for col in ["Shape", "XLA ms", "Flash ms", "Speedup", "Score matrix"]:
        table.add_column(col, justify="right" if col != "Shape" else "left")

    key = jax.random.PRNGKey(2)
    for label, B, H, S, D in ATTN_SHAPES:
        q = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)
        k = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)
        v = jax.random.normal(key, (B, H, S, D), dtype=jnp.bfloat16)

        xla_fn = functools.partial(jax.nn.dot_product_attention, scale=D ** -0.5)

        spec   = LayerSpec(
            op_type="flash_attention", M=S, N=S, K=D,
            seq_len=S, num_heads=H, head_dim=D,
            batch_size=B,
        )
        pal_fn = _make_pallas_fn(spec)

        xla_ms  = timed_ms(xla_fn, q, k, v)
        pal_ms  = timed_ms(pal_fn, q, k, v)
        speedup = xla_ms / pal_ms

        score_mb = B * H * S * S * 4 / (1024 ** 2)
        colour   = "green"  if speedup >= 1.3  else (
                   "yellow" if speedup >= 1.05 else "red")
        table.add_row(
            label,
            f"{xla_ms:.3f}",
            f"{pal_ms:.3f}",
            f"[{colour}]{speedup:.2f}×[/{colour}]",
            f"{score_mb:.1f} MB",
        )

    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold]TPU Kernel Factory — v2 Benchmark[/bold]")
    console.print(f"JAX {jax.__version__} | Device: {jax.devices()[0]}\n")

    console.rule("Section A — Standalone MatMul")
    run_section_a()

    console.rule("Section B — Fused MatMul+RMSNorm")
    run_section_b()

    console.rule("Section C — FlashAttention")
    run_section_c()

    console.rule("[bold green]Done[/bold green]")
