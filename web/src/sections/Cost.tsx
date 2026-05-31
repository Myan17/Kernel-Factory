import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import Reveal from "../components/Reveal";
import SectionHeading from "../components/SectionHeading";
import { BENCHMARKS } from "../lib/benchmarks";
import { TPU_PRICE_PER_HR, fmtUSD } from "../lib/costs";
import { TPU_LABELS, type TpuVersion } from "../lib/hardware";

const TPUS: TpuVersion[] = ["v4", "v5e", "v6e"];

export default function Cost() {
  return (
    <section id="cost" className="scroll-mt-20 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="The business case"
          title={<>What a verified kernel <span className="text-gradient">actually saves</span></>}
          subtitle={
            <>
              Raw speed isn't the whole story — the factory is ~0.95× vs XLA on average. The money
              is in two places: <b className="text-ink">it cuts compute where its kernel genuinely wins</b>,
              and <b className="text-ink">it eliminates the autotuning search bill entirely</b>. Both
              calculators below run on the real measured v5e numbers.
            </>
          }
        />

        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <Reveal>
            <ComputeCalculator />
          </Reveal>
          <Reveal delay={0.1}>
            <AutotuneCalculator />
          </Reveal>
        </div>

        <Reveal delay={0.15}>
          <p className="mx-auto mt-8 max-w-3xl text-center text-sm leading-relaxed text-muted">
            Plus the cost you can't put on one line: every generated kernel is numerically verified
            before it ships, so you never pay for a TPU job that crashed on a hallucinated kernel.
          </p>
        </Reveal>
      </div>
    </section>
  );
}

// ── Card A — public (XLA) vs custom compute bill, per op ─────────────────────
function ComputeCalculator() {
  const [spend, setSpend] = useState(100_000);
  const [opIdx, setOpIdx] = useState(2); // ffn_down — a measured win
  const [share, setShare] = useState(25);

  const row = BENCHMARKS[opIdx];
  const { publicSlice, customSlice, deltaMo, deltaYr, ratio, wins } = useMemo(() => {
    const slice = (spend * share) / 100;
    const ratio = row.pallasMs / row.xlaMs; // custom / public
    const custom = slice * ratio;
    const dMo = slice - custom; // +ve = saving
    return {
      publicSlice: slice,
      customSlice: custom,
      deltaMo: dMo,
      deltaYr: dMo * 12,
      ratio,
      wins: dMo >= 0,
    };
  }, [spend, share, row]);

  const customPct = Math.min(140, ratio * 100);

  return (
    <div className="flex h-full flex-col rounded-xl2 border border-line bg-surface p-6 shadow-soft">
      <h3 className="font-display text-xl font-bold">Public kernel vs custom kernel</h3>
      <p className="mt-1 text-sm text-muted">
        Apply the measured latency ratio for one op to the slice of your TPU bill it represents.
      </p>

      <div className="mt-5 space-y-4">
        <label className="block">
          <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">
            monthly TPU compute spend
          </span>
          <div className="flex items-center rounded-lg border border-line bg-canvas px-3">
            <span className="font-mono text-sm text-muted">$</span>
            <input
              type="number"
              min={0}
              value={spend}
              onChange={(e) => setSpend(Math.max(0, Number(e.target.value) || 0))}
              className="w-full bg-transparent px-2 py-2 font-mono text-sm text-ink outline-none"
            />
            <span className="font-mono text-xs text-muted">/mo</span>
          </div>
        </label>

        <label className="block">
          <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">
            operation
          </span>
          <select
            value={opIdx}
            onChange={(e) => setOpIdx(Number(e.target.value))}
            className="w-full rounded-lg border border-line bg-canvas px-2.5 py-2 font-mono text-xs text-ink outline-none focus:border-indigo"
          >
            {BENCHMARKS.map((b, i) => (
              <option key={i} value={i}>
                {b.op} ({b.shape}) — {b.speedup.toFixed(2)}× {b.win ? "✓ win" : ""}
              </option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="mb-1.5 flex justify-between font-mono text-xs uppercase tracking-wider text-muted">
            <span>share of compute in this op</span>
            <span className="text-ink">{share}%</span>
          </span>
          <input
            type="range"
            min={1}
            max={100}
            value={share}
            onChange={(e) => setShare(Number(e.target.value))}
            className="w-full accent-[rgb(var(--indigo))]"
          />
        </label>
      </div>

      {/* comparison bars */}
      <div className="mt-5 space-y-3">
        <CostBar label="Public · XLA" value={fmtUSD(publicSlice) + "/mo"} pct={100} tone="muted" />
        <CostBar
          label="Custom · factory"
          value={fmtUSD(customSlice) + "/mo"}
          pct={customPct}
          tone={wins ? "mint" : "coral"}
        />
      </div>

      <div className="mt-auto pt-5">
        <div
          className={`rounded-xl px-4 py-3 ${
            wins ? "bg-mint/10 ring-1 ring-mint/40" : "bg-coral/10 ring-1 ring-coral/40"
          }`}
        >
          <div className="font-mono text-xs text-muted">
            {wins ? "you save on this slice" : "this op costs more — keep XLA here"}
          </div>
          <div className={`font-display text-2xl font-extrabold ${wins ? "text-mint" : "text-coral"}`}>
            {wins ? "" : "+"}
            {fmtUSD(Math.abs(deltaMo))}/mo
            <span className="ml-2 text-base font-bold text-muted">
              ({fmtUSD(Math.abs(deltaYr))}/yr)
            </span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-muted">
            {wins
              ? "the factory's benchmark gate flags this kernel as a win → ship it"
              : "the factory's benchmark gate flags this → it tells you to ship XLA instead"}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Card B — autotuning search bill, eliminated ──────────────────────────────
function AutotuneCalculator() {
  const [tpu, setTpu] = useState<TpuVersion>("v5e");
  const [shapes, setShapes] = useState(40);
  const [trials, setTrials] = useState(5000);
  const [sec, setSec] = useState(0.9);

  const { hours, costMo, costYr } = useMemo(() => {
    const h = (shapes * trials * sec) / 3600;
    const c = h * TPU_PRICE_PER_HR[tpu];
    return { hours: h, costMo: c, costYr: c * 12 };
  }, [shapes, trials, sec, tpu]);

  return (
    <div className="flex h-full flex-col rounded-xl2 border border-line bg-surface p-6 shadow-soft">
      <h3 className="font-display text-xl font-bold">Autotuning bill, eliminated</h3>
      <p className="mt-1 text-sm text-muted">
        Autotuners search 10³–10⁵ tilings <i>per shape</i> on real TPUs. The factory computes the
        tile directly — that search compute goes to zero.
      </p>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <label className="col-span-2 block">
          <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">tuning hardware</span>
          <div className="flex gap-1 rounded-xl border border-line bg-canvas p-1">
            {TPUS.map((t) => (
              <button
                key={t}
                onClick={() => setTpu(t)}
                className={`flex-1 rounded-lg px-2 py-1.5 text-xs font-medium transition ${
                  tpu === t ? "bg-ink text-canvas shadow-soft" : "text-muted hover:text-ink"
                }`}
              >
                {TPU_LABELS[t]}
              </button>
            ))}
          </div>
        </label>
        <NumField label="new shapes / mo" value={shapes} onChange={setShapes} />
        <NumField label="trials / shape" value={trials} onChange={setTrials} step={500} />
        <NumField label="sec / trial" value={sec} onChange={setSec} step={0.1} float />
        <div className="flex flex-col justify-end">
          <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">@ list price</span>
          <div className="rounded-lg border border-line bg-canvas px-2.5 py-2 font-mono text-sm text-ink">
            {fmtUSD(TPU_PRICE_PER_HR[tpu])}/hr
          </div>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        <CostBar label="Autotuning search" value={`${Math.round(hours).toLocaleString()} TPU-hr/mo`} pct={100} tone="coral" />
        <CostBar label="Factory · constraint math" value="< 1 ms · ~$0" pct={1.5} tone="mint" />
      </div>

      <div className="mt-auto pt-5">
        <div className="rounded-xl bg-mint/10 px-4 py-3 ring-1 ring-mint/40">
          <div className="font-mono text-xs text-muted">search compute you stop paying for</div>
          <div className="font-display text-2xl font-extrabold text-mint">
            {fmtUSD(costMo)}/mo
            <span className="ml-2 text-base font-bold text-muted">({fmtUSD(costYr)}/yr)</span>
          </div>
          <div className="mt-1 font-mono text-[11px] text-muted">
            ≈ {Math.round(hours).toLocaleString()} TPU-hours of search, replaced by pure math
          </div>
        </div>
      </div>
    </div>
  );
}

// ── shared bits ──────────────────────────────────────────────────────────────
function CostBar({
  label,
  value,
  pct,
  tone,
}: {
  label: string;
  value: string;
  pct: number;
  tone: "muted" | "mint" | "coral";
}) {
  const bg =
    tone === "mint" ? "rgb(var(--mint))" : tone === "coral" ? "rgb(var(--coral))" : "rgb(var(--muted))";
  return (
    <div>
      <div className="mb-1 flex justify-between font-mono text-xs">
        <span className="text-muted">{label}</span>
        <span className="font-bold text-ink">{value}</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-canvas">
        <motion.div
          className="h-full rounded-full"
          style={{ background: bg }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, pct)}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step = 1,
  float,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  float?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted">{label}</span>
      <input
        type="number"
        min={0}
        step={step}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value);
          onChange(float ? Math.max(0, n || 0) : Math.max(0, Math.floor(n || 0)));
        }}
        className="w-full rounded-lg border border-line bg-canvas px-2.5 py-2 font-mono text-sm text-ink outline-none focus:border-indigo"
      />
    </label>
  );
}
