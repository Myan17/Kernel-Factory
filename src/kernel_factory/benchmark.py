"""Benchmark the factory's custom Pallas kernels against the XLA-native baseline.

For each (op, shape) it runs the *entire pipeline* (KernelPipeline.run -> solve,
RAG/template, assemble, CPU-verify), then on the active device:

  1. baseline: jax.jit of the native XLA op (the "open source / library" impl)
  2. custom:   jax.jit of pl.pallas_call with the solver's tiles (real Mosaic)

Both are compiled once, then timed in a warm loop. Custom output is checked
against the baseline for correctness. Reports latency, throughput, and speedup.

Runs on a TPU for real numbers (interpret=False); pass interpret=True to exercise
the harness on CPU without a TPU (timings are then meaningless, correctness holds).
"""
from __future__ import annotations

import time
from typing import Optional

import jax
import jax.numpy as jnp
from pydantic import BaseModel

from kernel_factory.pipeline import KernelPipeline
from kernel_factory.schemas import DType, HardwareLimits, KernelConfig, LayerSpec

# Same tolerances the verification gate uses: bf16 accumulation ~1% rel error.
_ATOL = 1e-1
_RTOL = 1e-2

_DEFAULT_ITERS = 50


def _jnp_dtype(d: DType):
    return {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "int8": jnp.int8}[d.value]


class BenchmarkResult(BaseModel):
    op_type: str
    M: int
    N: int
    K: int
    block_m: int
    block_n: int
    block_k: int
    baseline_ms: float          # XLA-native, per call
    custom_ms: float            # custom Pallas kernel, per call
    speedup: float              # baseline_ms / custom_ms  (>1 = custom faster)
    custom_compile_ms: float    # first-call compile of the custom kernel
    throughput: float           # achieved on the custom kernel
    throughput_unit: str        # "TFLOP/s" (matmul) | "GB/s" (rmsnorm)
    passed: bool                # custom matches baseline within tolerance
    max_abs_error: Optional[float] = None
    pipeline_passed: bool = False  # the factory pipeline's own CPU-verify result
    device: str = "unknown"


# ── timing ──────────────────────────────────────────────────────────────────

def _compile_ms(fn, *args) -> float:
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) * 1000


def _loop_ms(fn, *args, iters: int) -> float:
    out = fn(*args)
    jax.block_until_ready(out)  # ensure compiled/warm before timing
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1000


# ── matmul ────────────────────────────────────────────────────────────────--

def _matmul_inputs(spec: LayerSpec):
    in_dt = _jnp_dtype(spec.input_dtype)
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (spec.M, spec.K), dtype=in_dt)
    b = jax.random.normal(jax.random.fold_in(key, 1), (spec.K, spec.N), dtype=in_dt)
    return a, b


def _matmul_baseline(spec: LayerSpec):
    acc_dt = _jnp_dtype(spec.accumulator_dtype)
    out_dt = _jnp_dtype(spec.output_dtype)

    def fn(a, b):
        return jax.lax.dot_general(
            a.astype(acc_dt), b.astype(acc_dt), (([1], [0]), ([], [])),
            preferred_element_type=acc_dt,
        ).astype(out_dt)

    return jax.jit(fn)


def _matmul_custom(spec: LayerSpec, config: KernelConfig, interpret: bool):
    import jax.experimental.pallas as pl

    acc_dt = _jnp_dtype(spec.accumulator_dtype)
    out_dt = _jnp_dtype(spec.output_dtype)
    M, N, K = spec.M, spec.N, spec.K
    bm, bn = config.block_m, config.block_n

    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(
            a_ref[...].astype(acc_dt), b_ref[...].astype(acc_dt),
            preferred_element_type=acc_dt,
        ).astype(out_dt)

    def run(a, b):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), out_dt),
            grid=(M // bm, N // bn),
            in_specs=[
                pl.BlockSpec((bm, K), lambda m, n: (m, 0)),
                pl.BlockSpec((K, bn), lambda m, n: (0, n)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
            interpret=interpret,
        )(a, b)

    return jax.jit(run)


def _matmul_throughput(spec: LayerSpec, ms: float) -> float:
    flops = 2.0 * spec.M * spec.N * spec.K
    return flops / (ms / 1000.0) / 1e12  # TFLOP/s


# ── rmsnorm ─────────────────────────────────────────────────────────────────

def _rmsnorm_inputs(spec: LayerSpec):
    in_dt = _jnp_dtype(spec.input_dtype)
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (spec.M, spec.N), dtype=in_dt)
    w = jax.random.normal(jax.random.fold_in(key, 1), (spec.N,), dtype=in_dt)
    return x, w


def _rmsnorm_baseline(spec: LayerSpec):
    out_dt = _jnp_dtype(spec.output_dtype)

    def fn(x, w):
        x_f = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + 1e-6)
        return ((x_f / rms) * w.astype(jnp.float32)).astype(out_dt)

    return jax.jit(fn)


def _rmsnorm_custom(spec: LayerSpec, config: KernelConfig, interpret: bool):
    import jax.experimental.pallas as pl

    out_dt = _jnp_dtype(spec.output_dtype)
    M, N = spec.M, spec.N
    # RMSNorm reduces over the FULL feature dim, so the kernel must see a whole
    # row per block: keep block_n = N and tile only M. Tiling N would compute a
    # wrong per-block RMS regardless of what the solver picked for block_n.
    bm = config.block_m

    def kernel(x_ref, w_ref, o_ref):
        x_f = x_ref[...].astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + 1e-6)
        o_ref[...] = ((x_f / rms) * w_ref[...].astype(jnp.float32)).astype(out_dt)

    def run(x, w):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), out_dt),
            grid=(M // bm,),
            in_specs=[
                pl.BlockSpec((bm, N), lambda m: (m, 0)),
                pl.BlockSpec((N,), lambda m: (0,)),
            ],
            out_specs=pl.BlockSpec((bm, N), lambda m: (m, 0)),
            interpret=interpret,
        )(x, w)

    return jax.jit(run)


def _rmsnorm_throughput(spec: LayerSpec, ms: float) -> float:
    ib = spec.input_dtype.itemsize
    ob = spec.output_dtype.itemsize
    # read x, read w, write o
    bytes_moved = spec.M * spec.N * ib + spec.N * ib + spec.M * spec.N * ob
    return bytes_moved / (ms / 1000.0) / 1e9  # GB/s


# ── dispatch ──────────────────────────────────────────────────────────────--

_OPS = {
    "matmul": (_matmul_inputs, _matmul_baseline, _matmul_custom, _matmul_throughput, "TFLOP/s"),
    "rmsnorm": (_rmsnorm_inputs, _rmsnorm_baseline, _rmsnorm_custom, _rmsnorm_throughput, "GB/s"),
}


def benchmark_one(
    spec: LayerSpec,
    hw: HardwareLimits,
    *,
    interpret: bool = False,
    iters: int = _DEFAULT_ITERS,
) -> BenchmarkResult:
    if spec.op_type not in _OPS:
        raise ValueError(f"No benchmark for op_type={spec.op_type!r}")
    make_inputs, make_baseline, make_custom, throughput_of, unit = _OPS[spec.op_type]

    # 1. Run the entire factory pipeline (solve -> template -> assemble -> verify).
    pipe = KernelPipeline(hw=hw).run(spec)
    config = pipe.kernel_config

    # 2. Build baseline (XLA) and custom (Pallas) callables on the active device.
    inputs = make_inputs(spec)
    baseline_fn = make_baseline(spec)
    custom_fn = make_custom(spec, config, interpret)

    # 3. Correctness: custom vs the XLA baseline.
    base_out = baseline_fn(*inputs)
    cust_out = custom_fn(*inputs)
    jax.block_until_ready((base_out, cust_out))
    max_err = float(jnp.max(jnp.abs(
        cust_out.astype(jnp.float32) - base_out.astype(jnp.float32))))
    passed = bool(jnp.allclose(
        cust_out.astype(jnp.float32), base_out.astype(jnp.float32),
        atol=_ATOL, rtol=_RTOL))

    # 4. Time both: compile once, then a warm loop.
    custom_compile_ms = _compile_ms(custom_fn, *inputs)
    baseline_ms = _loop_ms(baseline_fn, *inputs, iters=iters)
    custom_ms = _loop_ms(custom_fn, *inputs, iters=iters)

    device = str(jax.devices()[0])

    return BenchmarkResult(
        op_type=spec.op_type, M=spec.M, N=spec.N, K=spec.K,
        block_m=config.block_m, block_n=config.block_n, block_k=config.block_k,
        baseline_ms=baseline_ms, custom_ms=custom_ms,
        speedup=baseline_ms / custom_ms,
        custom_compile_ms=custom_compile_ms,
        throughput=throughput_of(spec, custom_ms), throughput_unit=unit,
        passed=passed, max_abs_error=max_err,
        pipeline_passed=bool(pipe.test_result.passed),
        device=device,
    )


def run_benchmark(
    specs: list[LayerSpec],
    hw: HardwareLimits,
    *,
    interpret: bool = False,
    iters: int = _DEFAULT_ITERS,
) -> list[BenchmarkResult]:
    return [benchmark_one(s, hw, interpret=interpret, iters=iters) for s in specs]


def default_shapes() -> list[LayerSpec]:
    """A small sweep across both ops. All dims are multiples of 128."""
    matmuls = [(512, 512, 512), (1024, 1024, 512), (1024, 1024, 1024),
               (2048, 2048, 2048), (4096, 4096, 4096)]
    rmsnorms = [(1024, 1024), (4096, 4096), (8192, 4096)]
    specs = [LayerSpec(op_type="matmul", M=m, N=n, K=k) for (m, n, k) in matmuls]
    specs += [LayerSpec(op_type="rmsnorm", M=m, N=n, K=n) for (m, n) in rmsnorms]
    return specs


def format_table(results: list[BenchmarkResult]) -> str:
    header = (
        f"{'op':<8} {'M':>5} {'N':>5} {'K':>5} "
        f"{'baseline ms':>12} {'custom ms':>11} {'speedup':>8} "
        f"{'throughput':>12} {'unit':>8} {'ok':>3} {'max_err':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.op_type:<8} {r.M:>5} {r.N:>5} {r.K:>5} "
            f"{r.baseline_ms:>12.4f} {r.custom_ms:>11.4f} {r.speedup:>7.2f}x "
            f"{r.throughput:>12.2f} {r.throughput_unit:>8} "
            f"{('Y' if r.passed else 'N'):>3} "
            f"{(r.max_abs_error if r.max_abs_error is not None else float('nan')):>9.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    hw = HardwareLimits.for_v5e()
    print(f"Device: {jax.devices()}\n")
    results = run_benchmark(default_shapes(), hw)
    print(format_table(results))


if __name__ == "__main__":
    main()
