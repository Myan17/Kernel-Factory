# Kernel Factory — live demo site

A single-page marketing site **and** a genuinely interactive product demo for the
**TPU Kernel Factory**: a zero-hallucination pipeline that *computes* verified
JAX/Pallas kernels for Google TPUs instead of letting an LLM free-write them.

> The same kind of model that hallucinates broken kernels, turned into a machine
> that can only emit correct ones — because the numbers come from hardware math,
> not a guess.

## Quick start

```bash
cd web
npm install
npm run dev      # → http://localhost:5173
```

Other scripts:

```bash
npm run build    # type-check (tsc --noEmit) + production build to dist/
npm run preview  # serve the production build
npm run test     # Vitest — proves the ported solver matches the real Python
```

## What's real here (not faked)

The centerpiece **Live Demo** runs the **actual `TileSolver` algorithm**, ported
1:1 from `src/kernel_factory/solver.py` into `src/lib/solver.ts`. Pick a shape and
TPU and it computes the real tile `(block_m, block_n, block_k)` with pure
constraint math — no search — then assembles a runnable kernel from the verified
Pallas template via the same `template.format(...)` substitution the Python
Assembler uses.

Because it's a faithful port, the elimination animation is honest:

- **Alignment** crosses out `16/32/64` on the N/K axes every run (they fail the
  `% 128` vector-width rule).
- **VMEM budget** is light on v5e/v6e for GPT-2 shapes (they fit with headroom)
  but bites on **v4** — the *v4 squeeze* preset (2048³ on 16 MiB) drives the
  utilization gauge to exactly **75%**, the safety budget.

### Verifying faithfulness

`src/lib/solver.test.ts` asserts the TS port against ground truth captured from
the real Python solver:

```bash
# ground truth was generated with:
cd .. && .venv/bin/python -c "
from kernel_factory.schemas import LayerSpec, HardwareLimits
from kernel_factory.solver import TileSolver
c = TileSolver(HardwareLimits.for_v5e()).solve(LayerSpec(op_type='matmul', M=2048, N=2048, K=2048))
print(c.block_m, c.block_n, c.block_k, c.total_vmem_estimate_bytes)
"
# → 2048 2048 2048 41943040   (matches the Vitest expectation)
```

> Note: the build prompt's worked-example table (e.g. attn_qkv → 256/256) came
> from an older solver. The shipped `solver.py` maximizes tile volume within the
> 75% VMEM budget, so for these shapes it selects the **full dimensions**. This
> demo matches the shipped code — run the CLI and you'll see the same numbers.

The numbers shown in the **Verify** stage (`max_abs_error`, latency) are
clearly-labelled roofline *estimates* (`src/lib/estimate.ts`). The **Proof**
section uses the real measured TPU v5e benchmark dataset
(`src/lib/benchmarks.ts`) and leads with the honest framing: ~0.95× average vs
XLA, with targeted wins on RMSNorm and `ffn_down`.

## Stack

Vite · React · TypeScript · Tailwind CSS · Framer Motion · Recharts (lazy-loaded).
The force-directed knowledge graph and the Python syntax highlighter are
hand-rolled to keep dependencies lean. Dark mode, `prefers-reduced-motion`, and
mobile layouts are all supported.

## Structure

```
src/
  lib/        solver.ts (the 1:1 port) · hardware.ts · templates.ts ·
              benchmarks.ts · graph.ts · presets.ts · estimate.ts · highlight.ts
  state/      theme + shared pipeline state (demo ↔ architecture light up together)
  components/ CodeBlock · TileSolveViz · VmemGauge · ForceGraph · Terminal · Duck · …
  sections/   Hero · Problem · LiveDemo · Architecture · Proof · KnowledgeGraph · Tooling · Footer
```

No Python source is modified by this app.
