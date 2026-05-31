import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import SectionHeading from "../components/SectionHeading";
import { usePipeline } from "../state/pipeline";

const STAGES = [
  { key: "spec", name: "LayerSpec", sub: "M · N · K · dtype", tag: "input" },
  { key: "solve", name: "TileSolver", sub: "constraint math · no search", tag: "stage" },
  { key: "rag", name: "RAG · LanceDB", sub: "retrieve verified template", tag: "stage" },
  { key: "asm", name: "Assembler", sub: "inject solved integers", tag: "stage" },
  { key: "verify", name: "VerificationGate", sub: "jnp.allclose · 1e-2", tag: "stage" },
  { key: "kernel", name: "Runnable kernel", sub: "compiled & numerically proven", tag: "output" },
];

// stage boxes (indices 1..4) map to pipeline stages 0..3
const STAGE_TO_PIPELINE: Record<number, number> = { 1: 0, 2: 1, 3: 2, 4: 3 };

export default function Architecture() {
  const { activeStage, reached, running, runToken, verify } = usePipeline();
  const reduce = useReducedMotion();
  const [idle, setIdle] = useState(0);

  // gentle idle shimmer through the stages when the demo isn't running
  useEffect(() => {
    if (reduce || running || reached >= 0) return;
    const t = setInterval(() => setIdle((i) => (i + 1) % 6), 1100);
    return () => clearInterval(t);
  }, [reduce, running, reached]);

  const litState = (boxIdx: number): "active" | "done" | "idle" => {
    if (running || reached >= 0) {
      if (boxIdx === 0) return reached >= 0 ? "done" : "idle";
      if (boxIdx === 5) return reached >= 3 && verify?.status === "pass" ? "done" : "idle";
      const ps = STAGE_TO_PIPELINE[boxIdx];
      if (activeStage === ps) return "active";
      if (reached > ps) return "done";
      return "idle";
    }
    return idle === boxIdx ? "active" : "idle";
  };

  return (
    <section id="architecture" className="scroll-mt-20 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          center
          eyebrow="The pipeline"
          title={<>One path: <span className="text-gradient">spec → kernel</span></>}
          subtitle="The agent is an assembler, not a coder — it never writes a Pallas primitive from scratch. Run the demo above and these stages light up in lockstep."
        />

        <div className="relative mt-14">
          <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-center">
            {STAGES.map((s, i) => {
              const state = litState(i);
              return (
                <div key={s.key} className="flex flex-1 items-center gap-3 lg:flex-col">
                  <motion.div
                    key={`${s.key}-${runToken}`}
                    animate={{
                      scale: state === "active" ? 1.05 : 1,
                    }}
                    transition={{ duration: 0.3 }}
                    className={[
                      "relative w-full rounded-xl2 border p-4 text-center transition",
                      s.tag === "input"
                        ? "bg-duck/10"
                        : s.tag === "output"
                        ? "bg-mint/10"
                        : "bg-surface",
                      state === "active"
                        ? "border-indigo shadow-glow"
                        : state === "done"
                        ? "border-mint/50"
                        : "border-line",
                    ].join(" ")}
                  >
                    <div className="flex items-center justify-center gap-2">
                      {s.tag === "stage" && (
                        <span
                          className={`grid h-5 w-5 place-items-center rounded-md font-mono text-[10px] font-bold ${
                            state === "done"
                              ? "bg-mint/25 text-mint"
                              : state === "active"
                              ? "bg-indigo text-white"
                              : "bg-canvas text-muted"
                          }`}
                        >
                          {state === "done" ? "✓" : STAGE_TO_PIPELINE[i] + 1}
                        </span>
                      )}
                      <div className="font-display text-sm font-bold">{s.name}</div>
                    </div>
                    <div className="mt-1 font-mono text-[11px] text-muted">{s.sub}</div>
                    {state === "active" && (
                      <motion.span
                        layoutId="arch-pulse"
                        className="absolute -top-1.5 -right-1.5 h-3 w-3 rounded-full bg-indigo"
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      />
                    )}
                  </motion.div>
                  {i < STAGES.length - 1 && (
                    <div className="shrink-0 text-muted lg:rotate-0">
                      <span className="hidden lg:inline">→</span>
                      <span className="lg:hidden">↓</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* KG provenance strip */}
          <div className="mt-6 flex items-center justify-center">
            <div className="rounded-full border border-coral/40 bg-coral/10 px-4 py-2 text-center font-mono text-xs text-coral">
              ↑ Knowledge Graph · Kuzu — provenance of every spec, tile, kernel & result
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
