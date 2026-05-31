import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Reveal from "../components/Reveal";
import SectionHeading from "../components/SectionHeading";
import CountUp from "../components/CountUp";
import { BENCHMARKS, HEADLINE, avgSpeedup, type Benchmark } from "../lib/benchmarks";

type View = "speedup" | "latency" | "tflops";
type Kind = "all" | "matmul" | "rmsnorm" | "fused";

const INDIGO = "#5B5BD6";
const MINT = "#3DDC97";
const DUCK = "#FFC53D";
const MUTED = "#9396b3";

function ChartTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row: Benchmark = payload[0].payload;
  return (
    <div className="rounded-xl border border-line bg-surface px-3 py-2 font-mono text-xs shadow-lift">
      <div className="font-bold text-ink">{row.op}</div>
      <div className="text-muted">{row.shape}</div>
      <div className="mt-1 space-y-0.5">
        <div>XLA <span className="text-ink">{row.xlaMs.toFixed(3)} ms</span> · {row.xlaTflops.toFixed(2)} TF</div>
        <div>Pallas <span className="text-ink">{row.pallasMs.toFixed(3)} ms</span> · {row.pallasTflops.toFixed(2)} TF</div>
        <div className={row.win ? "text-mint" : "text-muted"}>speedup {row.speedup.toFixed(2)}× {row.win ? "✓ win" : ""}</div>
      </div>
    </div>
  );
}

export default function Proof() {
  const [view, setView] = useState<View>("speedup");
  const [kind, setKind] = useState<Kind>("all");

  const rows = useMemo(
    () => (kind === "all" ? BENCHMARKS : BENCHMARKS.filter((b) => b.kind === kind)),
    [kind]
  );
  const avg = useMemo(() => avgSpeedup(rows), [rows]);

  return (
    <section id="proof" className="scroll-mt-20 bg-surface/40 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="Proof"
          title={<>Measured on real <span className="text-gradient">TPU v5e</span> silicon</>}
          subtitle={
            <>
              The honest story: <b className="text-ink">~0.95× average vs XLA</b> — the win is
              correctness-guaranteed generation of fused/non-standard kernels XLA can't express, plus
              measured gains on RMSNorm (+3–4%) and asymmetric ffn_down (+5%). No blanket-speedup claim.
            </>
          }
        />

        {/* headline stats */}
        <div className="mt-10 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {[
            { v: <><CountUp to={10} />/10</>, l: "kernels compiled & ran", c: "text-mint" },
            { v: <><CountUp to={92} />/92</>, l: "unit tests pass", c: "text-mint" },
            { v: <CountUp to={0.95} decimals={2} suffix="×" />, l: "avg latency vs XLA", c: "text-indigo" },
            { v: <>+20–50%</>, l: "headroom (roadmap)", c: "text-duck" },
          ].map((s, i) => (
            <Reveal key={i} delay={i * 0.08}>
              <div className="rounded-xl2 border border-line bg-surface p-5 text-center shadow-soft">
                <div className={`font-display text-3xl font-extrabold tabular-nums sm:text-4xl ${s.c}`}>{s.v}</div>
                <div className="mt-1 text-xs text-muted">{s.l}</div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* chart card */}
        <Reveal delay={0.1}>
          <div className="mt-8 rounded-xl2 border border-line bg-surface p-5 shadow-soft sm:p-6">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <Toggle
                value={view}
                onChange={(v) => setView(v as View)}
                options={[
                  ["speedup", "Speedup"],
                  ["latency", "XLA vs Pallas (ms)"],
                  ["tflops", "TFLOPS"],
                ]}
              />
              <Toggle
                value={kind}
                onChange={(v) => setKind(v as Kind)}
                options={[
                  ["all", "All"],
                  ["matmul", "matmul"],
                  ["rmsnorm", "rmsnorm"],
                  ["fused", "fused"],
                ]}
              />
            </div>

            <div className="h-[360px] w-full text-muted" style={{ color: MUTED }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 48, left: 0 }} barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} vertical={false} />
                  <XAxis
                    dataKey="op"
                    tick={{ fill: "currentColor", fontSize: 11, fontFamily: "JetBrains Mono" }}
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                    height={60}
                    stroke="currentColor"
                    opacity={0.5}
                  />
                  <YAxis
                    tick={{ fill: "currentColor", fontSize: 11, fontFamily: "JetBrains Mono" }}
                    stroke="currentColor"
                    opacity={0.5}
                    width={48}
                  />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(91,91,214,0.08)" }} />
                  {view === "speedup" && (
                    <>
                      <ReferenceLine y={1} stroke={DUCK} strokeWidth={2} strokeDasharray="5 4" label={{ value: "1.0× XLA", fill: DUCK, fontSize: 11, position: "right" }} />
                      <Bar dataKey="speedup" radius={[5, 5, 0, 0]} maxBarSize={46}>
                        {rows.map((r, i) => (
                          <Cell key={i} fill={r.win ? MINT : INDIGO} />
                        ))}
                      </Bar>
                    </>
                  )}
                  {view === "latency" && (
                    <>
                      <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 12 }} />
                      <Bar name="XLA" dataKey="xlaMs" fill={MUTED} radius={[5, 5, 0, 0]} maxBarSize={26} />
                      <Bar name="Pallas" dataKey="pallasMs" fill={INDIGO} radius={[5, 5, 0, 0]} maxBarSize={26} />
                    </>
                  )}
                  {view === "tflops" && (
                    <>
                      <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 12 }} />
                      <Bar name="XLA" dataKey="xlaTflops" fill={MUTED} radius={[5, 5, 0, 0]} maxBarSize={26} />
                      <Bar name="Pallas" dataKey="pallasTflops" fill={MINT} radius={[5, 5, 0, 0]} maxBarSize={26} />
                    </>
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 font-mono text-xs text-muted">
              <span>
                avg over shown: <b className={avg >= 1 ? "text-mint" : "text-ink"}>{avg.toFixed(3)}×</b>
              </span>
              <span>{HEADLINE.device} · {HEADLINE.devicePod} · {HEADLINE.jaxVersion}</span>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function Toggle<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: [T, string][];
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded-xl border border-line bg-canvas p-1">
      {options.map(([v, label]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
            value === v ? "bg-ink text-canvas shadow-soft" : "text-muted hover:text-ink"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
