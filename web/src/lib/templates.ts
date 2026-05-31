// Verified Pallas skeleton templates. The Assembler does template.format(...)
// substituting solver integers into {placeholders} — it NEVER writes Pallas
// primitives from scratch. Zero hallucination: the agent is an assembler, not a
// coder. Stored verbatim so the browser emits a real, copyable kernel.

import type { KernelConfig, LayerSpec } from "./solver";

export const MATMUL_TEMPLATE = `import jax
import jax.numpy as jnp
import jax_pallas as pl
import jax_pallas.tpu_primitives as pltpu

def matmul_kernel(a_ref, b_ref, o_ref, acc_ref, *, block_k={block_k}):
    """Dense MatMul Pallas kernel — template-generated."""
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        a_ref[...].astype(jnp.{accumulator_dtype}),
        b_ref[...].astype(jnp.{accumulator_dtype}),
        preferred_element_type=jnp.{accumulator_dtype},
    )

    @pl.when(pl.program_id(2) == {num_k_tiles} - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(jnp.{output_dtype})

def run_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k = {block_m}, {block_n}, {block_k}
    num_k_tiles = K // block_k
    grid = (M // block_m, N // block_n, num_k_tiles)
    in_specs = [
        pl.BlockSpec((block_m, block_k), lambda m, n, k: (m, k)),
        pl.BlockSpec((block_k, block_n), lambda m, n, k: (k, n)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n, k: (m, n))
    scratch_specs = [pltpu.VMEM((block_m, block_n), jnp.{accumulator_dtype})]
    return pl.pallas_call(
        matmul_kernel, grid=grid, in_specs=in_specs,
        out_specs=out_specs, scratch_shapes=scratch_specs,
    )(a, b)`;

export const RMSNORM_TEMPLATE = `import jax
import jax.numpy as jnp
import jax_pallas as pl

EPS = 1e-6

def rmsnorm_kernel(x_ref, w_ref, o_ref):
    """RMSNorm Pallas kernel — template-generated."""
    x = x_ref[...].astype(jnp.{accumulator_dtype})
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS)
    normed = (x / rms).astype(jnp.{output_dtype})
    o_ref[...] = normed * w_ref[...].astype(jnp.{output_dtype})

def run_rmsnorm(x, w):
    M, N = x.shape
    block_m, block_n = {block_m}, {block_n}
    in_specs = [
        pl.BlockSpec((block_m, block_n), lambda m, n: (m, n)),
        pl.BlockSpec((block_n,), lambda m, n: (n,)),
    ]
    out_specs = pl.BlockSpec((block_m, block_n), lambda m, n: (m, n))
    return pl.pallas_call(
        rmsnorm_kernel, grid=(M // block_m, N // block_n),
        in_specs=in_specs, out_specs=out_specs,
    )(x, w)`;

export const TEMPLATES: Record<LayerSpec["opType"], string> = {
  matmul: MATMUL_TEMPLATE,
  rmsnorm: RMSNORM_TEMPLATE,
};

/** The set of placeholder tokens, in the order a reader meets them. */
export const PLACEHOLDERS = [
  "block_m",
  "block_n",
  "block_k",
  "M",
  "N",
  "K",
  "input_dtype",
  "output_dtype",
  "accumulator_dtype",
  "num_k_tiles",
] as const;

export function substitutions(
  spec: LayerSpec,
  config: KernelConfig
): Record<string, string> {
  return {
    block_m: String(config.blockM),
    block_n: String(config.blockN),
    block_k: String(config.blockK),
    M: String(spec.M),
    N: String(spec.N),
    K: String(spec.K),
    input_dtype: config.inputDtype,
    output_dtype: config.outputDtype,
    accumulator_dtype: config.accumulatorDtype,
    num_k_tiles: String(config.numKTiles),
  };
}

/** Mirror of the Python Assembler: template.format(**subs). */
export function assemble(spec: LayerSpec, config: KernelConfig): string {
  const tpl = TEMPLATES[spec.opType];
  const subs = substitutions(spec, config);
  return tpl.replace(/\{(\w+)\}/g, (m, key: string) =>
    key in subs ? subs[key] : m
  );
}
