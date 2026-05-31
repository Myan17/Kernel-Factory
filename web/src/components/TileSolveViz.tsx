import { motion } from "framer-motion";
import type { SolveResult, AxisBreakdown } from "../lib/solver";

const AXIS_LABEL: Record<string, string> = {
  m: "block_m",
  n: "block_n",
  k: "block_k",
};

interface Props {
  result: SolveResult;
  step: number; // 0 idle, 1 alignment cross-out, 2 vmem, 3 winner spotlight
}

function winnerFor(axis: AxisBreakdown, result: SolveResult): number | null {
  const c = result.config;
  if (!c) return null;
  return axis.axis === "m" ? c.blockM : axis.axis === "n" ? c.blockN : c.blockK;
}

export default function TileSolveViz({ result, step }: Props) {
  return (
    <div className="space-y-3">
      {result.axes.map((axis) => {
        const winner = winnerFor(axis, result);
        return (
          <div key={axis.axis} className="flex flex-wrap items-center gap-2">
            <div className="w-24 shrink-0 font-mono text-xs text-muted">
              {AXIS_LABEL[axis.axis]}
              <span className="ml-1 text-[10px] text-coral/70">{axis.alignmentLabel}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {axis.chips.map((chip) => {
                const rejected = !chip.kept; // alignment / out-of-range
                const isWinner = chip.value === winner;
                const crossed = step >= 1 && rejected;
                const spotlight = step >= 3 && isWinner;
                const dimmed = step >= 3 && !isWinner && !rejected;
                return (
                  <motion.span
                    key={`${axis.axis}-${chip.value}-${chip.isFullDim}`}
                    layout
                    animate={{
                      opacity: crossed ? 0.32 : dimmed ? 0.4 : 1,
                      scale: spotlight ? 1.12 : 1,
                    }}
                    transition={{ duration: 0.35 }}
                    className={[
                      "relative rounded-lg border px-2.5 py-1 font-mono text-xs tabular-nums",
                      spotlight
                        ? "border-transparent bg-gradient-to-br from-indigo to-violet text-white shadow-glow"
                        : crossed
                        ? "border-line bg-canvas text-muted line-through decoration-coral/70 decoration-2"
                        : "border-line bg-surface text-ink",
                      chip.isFullDim && !crossed ? "ring-1 ring-duck/60" : "",
                    ].join(" ")}
                  >
                    {chip.value}
                    {chip.isFullDim && (
                      <span className="ml-1 text-[9px] text-duck">full</span>
                    )}
                    {spotlight && <span className="ml-1">★</span>}
                  </motion.span>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* running commentary */}
      <div className="pt-1 font-mono text-[11px] leading-relaxed text-muted">
        {step >= 1 && (
          <div>
            <span className="text-coral">✗ alignment</span> · cross out blocks not
            multiples of 128 (N/K) or 8 (M), and any block larger than its dim
          </div>
        )}
        {step >= 2 && (
          <div>
            <span className={result.eliminatedByVmem > 0 ? "text-coral" : "text-mint"}>
              {result.eliminatedByVmem > 0 ? "✗ VMEM" : "✓ VMEM"}
            </span>{" "}
            · {result.eliminatedByVmem > 0
              ? `${result.eliminatedByVmem} of ${result.totalCombos} aligned combos exceed the budget`
              : `all ${result.totalCombos} aligned combos fit — pick the largest`}
          </div>
        )}
        {step >= 3 && result.config && (
          <div>
            <span className="text-mint">★ chosen</span> · maximize block volume →{" "}
            <span className="text-ink">
              {result.config.blockM}×{result.config.blockN}×{result.config.blockK}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
