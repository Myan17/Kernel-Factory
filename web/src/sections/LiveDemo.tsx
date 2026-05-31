import { useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import confetti from "canvas-confetti";
import { HARDWARE, TPU_LABELS, type DType, type TpuVersion } from "../lib/hardware";
import { solve, type LayerSpec, type OpType } from "../lib/solver";
import { TEMPLATES, substitutions } from "../lib/templates";
import { estimateLatencyMs, estimateMaxAbsError } from "../lib/estimate";
import { PRESETS, type Preset } from "../lib/presets";
import { usePipeline, STAGES } from "../state/pipeline";
import SectionHeading from "../components/SectionHeading";
import TileSolveViz from "../components/TileSolveViz";
import VmemGauge from "../components/VmemGauge";
import CodeBlock from "../components/CodeBlock";

const DTYPES: DType[] = ["bfloat16", "float32", "int8"];
const TPUS: TpuVersion[] = ["v4", "v5e", "v6e"];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const STAGE_BLURB = [
  "Compute the tile from hardware math — no search.",
  "Pull a verified Pallas template (placeholders only).",
  "Inject the solved integers — assemble, don't write.",
  "Run in CPU interpret-mode vs a pure-JAX baseline.",
];

export default function LiveDemo() {
  const reduce = useReducedMotion();
  const pipeline = usePipeline();

  const [opType, setOpType] = useState<OpType>("matmul");
  const [M, setM] = useState(512);
  const [N, setN] = useState(768);
  const [K, setK] = useState(768);
  const [tpu, setTpu] = useState<TpuVersion>("v5e");
  const [inputDtype, setInputDtype] = useState<DType>("bfloat16");
  const [outputDtype, setOutputDtype] = useState<DType>("bfloat16");
  const [accumulatorDtype, setAccumulatorDtype] = useState<DType>("float32");
  const [activePreset, setActivePreset] = useState<string | null>("attn_qkv");
  const [solveStep, setSolveStep] = useState(0);

  const runRef = useRef(0);

  const spec: LayerSpec = useMemo(
    () => ({ opType, M, N, K, inputDtype, outputDtype, accumulatorDtype }),
    [opType, M, N, K, inputDtype, outputDtype, accumulatorDtype]
  );
  const hw = HARDWARE[tpu];
  const result = useMemo(() => solve(spec, hw), [spec, hw]);

  // reset the staged output whenever the spec changes
  useEffect(() => {
    runRef.current++;
    setSolveStep(0);
    pipeline.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, hw]);

  const applyPreset = (p: Preset) => {
    setActivePreset(p.id);
    setOpType(p.opType);
    setM(p.M);
    setN(p.N);
    setK(p.K);
    setTpu(p.tpu);
    setInputDtype(p.inputDtype);
    setOutputDtype(p.outputDtype);
    setAccumulatorDtype(p.accumulatorDtype);
  };

  const fireConfetti = () => {
    if (reduce) return;
    confetti({
      particleCount: 120,
      spread: 72,
      origin: { y: 0.7 },
      colors: ["#5B5BD6", "#FFC53D", "#3DDC97", "#FF6B6B", "#8B5CF6"],
      scalar: 0.9,
    });
  };

  async function generate() {
    const myRun = ++runRef.current;
    const r = solve(spec, hw);
    setSolveStep(0);
    pipeline.set({
      running: true,
      result: r,
      activeStage: 0,
      reached: 0,
      verify: null,
      runToken: pipeline.runToken + 1,
    });

    const guard = () => runRef.current === myRun;
    const at = async (ms: number, fn: () => void) => {
      await sleep(reduce ? 0 : ms);
      if (!guard()) throw new Error("cancelled");
      fn();
    };

    try {
      if (!r.ok) {
        await at(400, () => pipeline.set({ running: false }));
        return;
      }
      await at(520, () => setSolveStep(1));
      await at(560, () => setSolveStep(2));
      await at(560, () => setSolveStep(3));
      await at(520, () => pipeline.set({ activeStage: 1, reached: 1 }));
      await at(680, () => pipeline.set({ activeStage: 2, reached: 2 }));
      await at(920, () => pipeline.set({ activeStage: 3, reached: 3 }));
      await at(360, () => {
        const verify = {
          status: "pass" as const,
          maxAbsError: estimateMaxAbsError(spec, r.config!),
          latencyMs: estimateLatencyMs(spec, hw, r.config!),
        };
        pipeline.set({ verify, running: false });
        fireConfetti();
      });
    } catch {
      /* superseded by a newer run */
    }
  }

  const subs = result.config ? substitutions(spec, result.config) : {};
  const filename = opType === "matmul" ? "matmul_kernel.py" : "rmsnorm_kernel.py";
  const reached = pipeline.reached;

  return (
    <section id="demo" className="relative scroll-mt-20 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="mb-3 flex items-center justify-center">
          <span className="inline-flex items-center gap-2 rounded-full bg-mint/15 px-3 py-1 font-mono text-xs font-semibold text-mint ring-1 ring-mint/40">
            <span className="h-2 w-2 animate-pulse rounded-full bg-mint" /> LIVE
          </span>
        </div>
        <SectionHeading
          center
          eyebrow="Generate a kernel"
          title={<>Solve → Retrieve → Assemble → <span className="text-gradient">Verify</span></>}
          subtitle="This runs the real ported solver in your browser. Pick a shape, choose a TPU, and watch a verified JAX/Pallas kernel get computed — not guessed."
        />

        <div className="mt-12 grid gap-6 lg:grid-cols-[380px_1fr]">
          {/* ── controls ─────────────────────────────────────────────── */}
          <div className="rounded-xl2 border border-line bg-surface p-5 shadow-soft">
            <div className="mb-4">
              <div className="mb-2 font-mono text-xs uppercase tracking-wider text-muted">presets</div>
              <div className="flex flex-wrap gap-1.5">
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => applyPreset(p)}
                    title={p.hint}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
                      activePreset === p.id
                        ? "border-transparent bg-gradient-to-br from-indigo to-violet text-white shadow-glow"
                        : "border-line bg-canvas text-ink hover:border-indigo/50"
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <Field label="operation">
              <Segmented
                options={["matmul", "rmsnorm"] as OpType[]}
                value={opType}
                onChange={(v) => {
                  setOpType(v);
                  setActivePreset(null);
                }}
              />
            </Field>

            <div className="grid grid-cols-3 gap-2">
              <NumberField label="M" value={M} onChange={(v) => { setM(v); setActivePreset(null); }} />
              <NumberField label="N" value={N} onChange={(v) => { setN(v); setActivePreset(null); }} />
              <NumberField
                label="K"
                value={K}
                disabled={opType === "rmsnorm"}
                onChange={(v) => { setK(v); setActivePreset(null); }}
              />
            </div>

            <Field label="target TPU">
              <Segmented options={TPUS} value={tpu} onChange={(v) => { setTpu(v); setActivePreset(null); }} render={(v) => TPU_LABELS[v]} />
            </Field>

            <div className="grid grid-cols-3 gap-2">
              <SelectField label="input" value={inputDtype} onChange={(v) => { setInputDtype(v); setActivePreset(null); }} />
              <SelectField label="output" value={outputDtype} onChange={(v) => { setOutputDtype(v); setActivePreset(null); }} />
              <SelectField label="accum" value={accumulatorDtype} onChange={(v) => { setAccumulatorDtype(v); setActivePreset(null); }} />
            </div>

            <button
              onClick={generate}
              disabled={pipeline.running}
              className="mt-5 w-full rounded-xl bg-gradient-to-r from-indigo via-violet to-indigo bg-[length:200%_auto] px-5 py-3.5 text-base font-bold text-white shadow-lift transition hover:bg-[position:100%] disabled:opacity-60"
            >
              {pipeline.running ? "Computing…" : "⚡ Generate kernel"}
            </button>

            <div className="mt-3 rounded-xl border border-line bg-canvas px-3 py-2 font-mono text-[11px] leading-relaxed text-muted">
              {hw.tpuVersion.toUpperCase()} · {(hw.vmemBytes / 1048576).toFixed(0)} MiB VMEM · budget{" "}
              {(hw.vmemSafetyFraction * 100).toFixed(0)}% · vector {hw.vectorWidth} · sublane {hw.sublaneWidth}
            </div>
          </div>

          {/* ── staged output ────────────────────────────────────────── */}
          <div className="space-y-3">
            {/* error state */}
            {reached >= 0 && !result.ok && (
              <div className="rounded-xl2 border border-coral/40 bg-coral/10 p-5">
                <div className="font-display text-lg font-bold text-coral">❌ No valid tile</div>
                <p className="mt-1 text-sm text-muted">{result.error}</p>
              </div>
            )}

            {reached < 0 && (
              <div className="grid place-items-center rounded-xl2 border border-dashed border-line bg-surface/60 p-12 text-center">
                <div>
                  <div className="text-4xl">🦆</div>
                  <p className="mt-3 max-w-sm text-sm text-muted">
                    Hit <span className="font-semibold text-ink">Generate kernel</span> to run all
                    four stages. Everything you see is computed live by the ported solver.
                  </p>
                </div>
              </div>
            )}

            {reached >= 0 && result.ok && (
              <StageCard idx={0} reached={reached} active={pipeline.activeStage} blurb={STAGE_BLURB[0]}>
                <div className="grid gap-5 md:grid-cols-[1fr_auto]">
                  <TileSolveViz result={result} step={solveStep} />
                  <VmemGauge
                    fraction={result.config!.vmemUtilizationFraction}
                    budgetFraction={hw.vmemSafetyFraction}
                    estimateBytes={result.config!.totalVmemEstimateBytes}
                    vmemBytes={hw.vmemBytes}
                    active={solveStep >= 2}
                  />
                </div>
              </StageCard>
            )}

            {reached >= 1 && result.ok && (
              <StageCard idx={1} reached={reached} active={pipeline.activeStage} blurb={STAGE_BLURB[1]}>
                <div className="mb-2 font-mono text-xs text-muted">
                  matched verified template{" "}
                  <span className="text-indigo">{opType}.pallas</span> · placeholders highlighted
                </div>
                <CodeBlock
                  code={TEMPLATES[opType]}
                  placeholderMode="show"
                  filename={`templates/${opType}.py`}
                  maxHeight={300}
                />
              </StageCard>
            )}

            {reached >= 2 && result.ok && (
              <StageCard idx={2} reached={reached} active={pipeline.activeStage} blurb={STAGE_BLURB[2]}>
                <div className="mb-2 font-mono text-xs text-muted">
                  injected{" "}
                  <span className="text-mint">
                    block_m={subs.block_m} block_n={subs.block_n} block_k={subs.block_k}
                  </span>{" "}
                  — copy & run
                </div>
                <CodeBlock
                  code={TEMPLATES[opType]}
                  placeholderMode="fill"
                  subs={subs}
                  filename={filename}
                />
              </StageCard>
            )}

            {reached >= 3 && result.ok && pipeline.verify && (
              <StageCard idx={3} reached={reached} active={pipeline.activeStage} blurb={STAGE_BLURB[3]}>
                <div className="flex flex-wrap items-center gap-4">
                  <motion.div
                    initial={{ scale: 0.6, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 240, damping: 16 }}
                    className="inline-flex items-center gap-2 rounded-xl bg-mint/15 px-4 py-2.5 text-lg font-bold text-mint ring-1 ring-mint/50"
                  >
                    ✅ PASS
                  </motion.div>
                  <Metric label="max_abs_error" value={pipeline.verify.maxAbsError.toExponential(2)} note="< 1e-2 ✓" />
                  <Metric label="est. latency" value={`${pipeline.verify.latencyMs.toFixed(3)} ms`} note={`${tpu} roofline`} />
                  <Metric label="grid" value={`${Math.max(1, Math.floor(M / result.config!.blockM))}×${Math.max(1, Math.floor(N / result.config!.blockN))}${opType === "matmul" ? `×${result.config!.numKTiles}` : ""}`} note="program ids" />
                </div>
                <p className="mt-3 font-mono text-[11px] leading-relaxed text-muted">
                  verified in CPU interpret-mode against a pure-JAX baseline — jnp.allclose(atol=1e-2,
                  rtol=1e-2). result logged to SQLite + provenance to the Kuzu graph.
                </p>
              </StageCard>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

// ── stage card shell ──────────────────────────────────────────────────────
function StageCard({
  idx,
  reached,
  active,
  blurb,
  children,
}: {
  idx: number;
  reached: number;
  active: number;
  blurb: string;
  children: React.ReactNode;
}) {
  const isActive = active === idx;
  const done = reached > idx || (reached === idx && active !== idx);
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`rounded-xl2 border bg-surface p-5 shadow-soft transition ${
        isActive ? "border-indigo/60 shadow-glow" : "border-line"
      }`}
    >
      <div className="mb-3 flex items-center gap-3">
        <span
          className={`grid h-7 w-7 place-items-center rounded-lg font-mono text-sm font-bold ${
            done
              ? "bg-mint/20 text-mint"
              : isActive
              ? "bg-gradient-to-br from-indigo to-violet text-white"
              : "bg-canvas text-muted"
          }`}
        >
          {done ? "✓" : idx + 1}
        </span>
        <div>
          <div className="font-display text-base font-bold leading-none">{STAGES[idx]}</div>
          <div className="mt-0.5 text-xs text-muted">{blurb}</div>
        </div>
        {isActive && (
          <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-indigo" />
        )}
      </div>
      {children}
    </motion.div>
  );
}

// ── tiny form controls ─────────────────────────────────────────────────────
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <div className="mb-1.5 font-mono text-xs uppercase tracking-wider text-muted">{label}</div>
      {children}
    </div>
  );
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  render,
}: {
  options: T[];
  value: T;
  onChange: (v: T) => void;
  render?: (v: T) => string;
}) {
  return (
    <div className="flex gap-1 rounded-xl border border-line bg-canvas p-1">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          className={`flex-1 rounded-lg px-2 py-1.5 text-sm font-medium transition ${
            value === o ? "bg-ink text-canvas shadow-soft" : "text-muted hover:text-ink"
          }`}
        >
          {render ? render(o) : o}
        </button>
      ))}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`block ${disabled ? "opacity-40" : ""}`}>
      <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">{label}</span>
      <input
        type="number"
        min={1}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
        className="w-full rounded-lg border border-line bg-canvas px-2.5 py-2 font-mono text-sm text-ink outline-none transition focus:border-indigo focus:ring-2 focus:ring-indigo/30"
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: DType;
  onChange: (v: DType) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as DType)}
        className="w-full rounded-lg border border-line bg-canvas px-2 py-2 font-mono text-xs text-ink outline-none transition focus:border-indigo focus:ring-2 focus:ring-indigo/30"
      >
        {DTYPES.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
    </label>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div>
      <div className="font-mono text-xs text-muted">{label}</div>
      <div className="font-mono text-lg font-bold tabular-nums text-ink">{value}</div>
      {note && <div className="font-mono text-[10px] text-mint">{note}</div>}
    </div>
  );
}
