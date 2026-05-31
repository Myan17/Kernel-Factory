import { motion } from "framer-motion";
import { formatBytes } from "../lib/hardware";

interface Props {
  fraction: number; // utilization of total VMEM (0..1)
  budgetFraction: number; // safety budget as fraction of total (0.75)
  estimateBytes: number;
  vmemBytes: number;
  active: boolean; // animate the fill in
}

export default function VmemGauge({
  fraction,
  budgetFraction,
  estimateBytes,
  vmemBytes,
  active,
}: Props) {
  const pct = Math.min(100, fraction * 100);
  const near = fraction > budgetFraction * 0.92;
  const fill = near ? "var(--coral)" : "var(--mint)";

  return (
    <div className="flex items-stretch gap-3">
      <div className="relative h-44 w-16 overflow-hidden rounded-xl border border-line bg-canvas">
        {/* budget line */}
        <div
          className="absolute inset-x-0 z-10 border-t-2 border-dashed border-coral/70"
          style={{ bottom: `${budgetFraction * 100}%` }}
        >
          <span className="absolute -top-4 right-0.5 font-mono text-[9px] text-coral">
            budget
          </span>
        </div>
        <motion.div
          className="absolute inset-x-0 bottom-0"
          style={{ background: `rgb(${fill === "var(--coral)" ? "var(--coral)" : "var(--mint)"})` }}
          initial={{ height: 0 }}
          animate={{ height: active ? `${pct}%` : 0 }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="h-full w-full opacity-90" style={{ background: `linear-gradient(0deg, rgb(${fill}), rgb(var(--indigo)))` }} />
        </motion.div>
      </div>
      <div className="flex flex-col justify-between py-1">
        <div>
          <div className="font-mono text-2xl font-bold tabular-nums text-ink">
            {pct.toFixed(pct < 10 ? 2 : 1)}<span className="text-base text-muted">%</span>
          </div>
          <div className="text-xs text-muted">VMEM utilization</div>
        </div>
        <div className="space-y-0.5 font-mono text-[11px] text-muted">
          <div>est&nbsp;&nbsp;{formatBytes(estimateBytes)}</div>
          <div>total {formatBytes(vmemBytes)}</div>
          <div className="text-coral/80">budget {(budgetFraction * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  );
}
