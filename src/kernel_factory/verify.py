"""
Self-contained verification harness.
Safe to ship via SSH to a TPU VM — only standard deps (jax, numpy, sqlite3).
"""
from __future__ import annotations

import enum
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from kernel_factory.schemas import DType, KernelConfig, LayerSpec, TestResult


class VerifyMode(str, enum.Enum):
    CPU_INTERPRET = "cpu_interpret"
    TPU_INTERPRET = "tpu_interpret"
    TPU_COMPILE = "tpu_compile"


# ── JAX baselines ─────────────────────────────────────────────────────────────

def _baseline_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.dot_general(a, b, (([1], [0]), ([], [])))


def _baseline_rmsnorm(x: jnp.ndarray, w: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_f = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + eps)
    return ((x_f / rms) * w.astype(jnp.float32)).astype(x.dtype)


# ── dtype helpers ─────────────────────────────────────────────────────────────

def _jnp_dtype(d: DType):
    return {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "int8": jnp.int8}[d.value]


# ── CPU interpret runners ─────────────────────────────────────────────────────

def _run_matmul_cpu(
    spec: LayerSpec, config: KernelConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    import jax.experimental.pallas as pl

    bm, bn = config.block_m, config.block_n
    M, N, K = spec.M, spec.N, spec.K
    in_dt = _jnp_dtype(spec.input_dtype)
    out_dt = _jnp_dtype(spec.output_dtype)
    acc_dt = _jnp_dtype(spec.accumulator_dtype)

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=in_dt)
    b = jax.random.normal(jax.random.fold_in(key, 1), (K, N), dtype=in_dt)

    # CPU interpret: reduce full K in one shot per tile — no scratch needed.
    # The actual TPU kernel tiles K; this verifies numerical correctness only.
    def kernel(a_ref, b_ref, o_ref):
        o_ref[...] = jnp.dot(
            a_ref[...].astype(acc_dt),
            b_ref[...].astype(acc_dt),
            preferred_element_type=acc_dt,
        ).astype(out_dt)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), out_dt),
        grid=(M // bm, N // bn),
        in_specs=[
            pl.BlockSpec((bm, K), lambda m, n: (m, 0)),
            pl.BlockSpec((K, bn), lambda m, n: (0, n)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
        interpret=True,
    )(a, b)

    baseline = _baseline_matmul(a.astype(acc_dt), b.astype(acc_dt)).astype(out_dt)
    return result, baseline


def _run_rmsnorm_cpu(
    spec: LayerSpec, config: KernelConfig
) -> tuple[jnp.ndarray, jnp.ndarray]:
    import jax.experimental.pallas as pl

    bm, bn = config.block_m, config.block_n
    M, N = spec.M, spec.N
    in_dt = _jnp_dtype(spec.input_dtype)
    out_dt = _jnp_dtype(spec.output_dtype)
    acc_dt = _jnp_dtype(spec.accumulator_dtype)

    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (M, N), dtype=in_dt)
    w = jax.random.normal(jax.random.fold_in(key, 1), (N,), dtype=in_dt)

    def kernel(x_ref, w_ref, o_ref):
        x_f = x_ref[...].astype(acc_dt)
        rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + 1e-6)
        o_ref[...] = ((x_f / rms) * w_ref[...].astype(acc_dt)).astype(out_dt)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), out_dt),
        grid=(M // bm, N // bn),
        in_specs=[
            pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
            pl.BlockSpec((bn,), lambda m, n: (n,)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
        interpret=True,
    )(x, w)

    baseline = _baseline_rmsnorm(x, w)
    return result, baseline


_CPU_RUNNERS = {
    "matmul": _run_matmul_cpu,
    "rmsnorm": _run_rmsnorm_cpu,
}

# Tolerances: bfloat16 accumulation has ~1% relative error vs float32 baseline
_ATOL = {"matmul": 1e-1, "rmsnorm": 1e-1}
_RTOL = {"matmul": 1e-2, "rmsnorm": 1e-2}


# ── SQLite logging ─────────────────────────────────────────────────────────────

def _ensure_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kernel_results (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            op_type              TEXT,
            block_m              INTEGER,
            block_n              INTEGER,
            block_k              INTEGER,
            M                    INTEGER,
            N                    INTEGER,
            K                    INTEGER,
            input_dtype          TEXT,
            output_dtype         TEXT,
            passed               INTEGER,
            max_abs_error        REAL,
            compile_time_ms      REAL,
            execution_latency_ms REAL,
            error_trace          TEXT,
            tpu_version          TEXT,
            created_at           TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _log(conn: sqlite3.Connection, spec: LayerSpec, config: KernelConfig, r: TestResult):
    conn.execute("""
        INSERT INTO kernel_results (
            op_type, block_m, block_n, block_k, M, N, K,
            input_dtype, output_dtype, passed, max_abs_error,
            compile_time_ms, execution_latency_ms, error_trace, tpu_version
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        spec.op_type, config.block_m, config.block_n, config.block_k,
        spec.M, spec.N, spec.K,
        spec.input_dtype.value, spec.output_dtype.value,
        int(r.passed), r.max_abs_error,
        r.compile_time_ms, r.execution_latency_ms,
        r.error_trace, r.tpu_version,
    ))
    conn.commit()


# ── Public gate ───────────────────────────────────────────────────────────────

class VerificationGate:
    def __init__(
        self,
        spec: LayerSpec,
        config: KernelConfig,
        mode: VerifyMode = VerifyMode.CPU_INTERPRET,
        db_path: Optional[Path] = None,
    ):
        self.spec = spec
        self.config = config
        self.mode = mode
        self.db_path = db_path

    def run(self) -> TestResult:
        spec, config = self.spec, self.config

        if self.mode != VerifyMode.CPU_INTERPRET:
            return TestResult(
                kernel_config=config, layer_spec=spec, passed=False,
                error_trace=f"VerifyMode {self.mode} requires a TPU VM — run verify.py remotely.",
            )

        runner = _CPU_RUNNERS.get(spec.op_type)
        if runner is None:
            return TestResult(
                kernel_config=config, layer_spec=spec, passed=False,
                error_trace=f"No CPU runner for op_type='{spec.op_type}'",
            )

        t0 = time.perf_counter()
        try:
            actual, expected = runner(spec, config)
            latency_ms = (time.perf_counter() - t0) * 1000
            max_err = float(jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))))
            passed = bool(jnp.allclose(
                actual.astype(jnp.float32),
                expected.astype(jnp.float32),
                atol=_ATOL[spec.op_type],
                rtol=_RTOL[spec.op_type],
            ))
            result = TestResult(
                kernel_config=config, layer_spec=spec,
                passed=passed,
                max_abs_error=max_err,
                execution_latency_ms=latency_ms,
            )
        except Exception:
            result = TestResult(
                kernel_config=config, layer_spec=spec,
                passed=False,
                error_trace=traceback.format_exc(),
            )

        if self.db_path:
            conn = _ensure_db(self.db_path)
            _log(conn, spec, config, result)
            conn.close()

        return result
