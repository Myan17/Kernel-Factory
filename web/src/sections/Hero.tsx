import { useEffect, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import TileField from "../components/TileField";
import { HEADLINE } from "../lib/benchmarks";

const STAGES = ["Solve", "Retrieve", "Assemble", "Verify"];
const REPO = "https://github.com/Myan17/QuackHacks";

function PipelineTyper() {
  const reduce = useReducedMotion();
  const [stage, setStage] = useState(0);
  const [len, setLen] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (reduce) return;
    const word = STAGES[stage];
    let delay = deleting ? 45 : 95;
    if (!deleting && len === word.length) delay = 1100;
    if (deleting && len === 0) delay = 220;
    const t = setTimeout(() => {
      if (!deleting && len < word.length) setLen(len + 1);
      else if (!deleting && len === word.length) setDeleting(true);
      else if (deleting && len > 0) setLen(len - 1);
      else {
        setDeleting(false);
        setStage((s) => (s + 1) % STAGES.length);
      }
    }, delay);
    return () => clearTimeout(t);
  }, [len, deleting, stage, reduce]);

  if (reduce) {
    return <span className="text-gradient">{STAGES.join(" → ")}</span>;
  }

  return (
    <span className="text-gradient">
      {STAGES.map((s, i) => (
        <span key={s}>
          {i < stage ? s : i === stage ? s.slice(0, len) : ""}
          {i === stage && <span className="ml-0.5 inline-block w-[2px] animate-pulse bg-current align-middle" style={{ height: "0.9em" }} />}
          {i < stage && <span className="mx-1.5 text-muted">→</span>}
        </span>
      ))}
    </span>
  );
}

export default function Hero() {
  return (
    <section id="top" className="relative isolate overflow-hidden pt-28 sm:pt-36">
      <div className="absolute inset-0 -z-10 bg-tiles opacity-40" />
      <div className="absolute inset-0 -z-10">
        <TileField />
      </div>
      <div
        className="absolute inset-x-0 top-0 -z-10 h-[520px] opacity-70"
        style={{
          background:
            "radial-gradient(60% 60% at 50% 0%, rgb(var(--indigo) / 0.18), transparent 70%)",
        }}
      />

      <div className="mx-auto max-w-5xl px-4 pb-20 text-center sm:px-6 sm:pb-28">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-line bg-surface/80 px-3.5 py-1.5 font-mono text-xs text-muted backdrop-blur"
        >
          🦆 Built at QuackHacks · zero-hallucination TPU kernels
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="font-display text-5xl font-extrabold leading-[1.02] tracking-tight sm:text-7xl"
        >
          Kernel<span className="text-gradient">Factory</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="mx-auto mt-5 max-w-2xl text-balance text-lg leading-relaxed text-muted sm:text-xl"
        >
          Verified JAX/Pallas kernels for Google TPUs — <span className="font-semibold text-ink">computed, not guessed.</span>{" "}
          The same kind of model that hallucinates broken kernels, turned into a machine that can only emit correct ones.
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="mt-7 font-mono text-2xl font-bold tracking-tight sm:text-3xl"
        >
          <PipelineTyper />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="mt-9 flex flex-wrap items-center justify-center gap-3"
        >
          <a
            href="#demo"
            className="rounded-xl bg-gradient-to-r from-indigo via-violet to-indigo bg-[length:200%_auto] px-6 py-3.5 text-base font-bold text-white shadow-lift transition hover:bg-[position:100%]"
          >
            ⚡ Try the live demo
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="rounded-xl border border-line bg-surface px-6 py-3.5 text-base font-semibold text-ink transition hover:border-indigo/50"
          >
            View on GitHub ↗
          </a>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5 }}
          className="mx-auto mt-12 flex max-w-2xl flex-wrap items-center justify-center gap-x-8 gap-y-3 font-mono text-sm text-muted"
        >
          <span><b className="text-mint">{HEADLINE.kernelsRan}/{HEADLINE.kernelsTotal}</b> kernels ran</span>
          <span className="hidden h-4 w-px bg-line sm:block" />
          <span><b className="text-mint">{HEADLINE.testsPass}/{HEADLINE.testsTotal}</b> tests pass</span>
          <span className="hidden h-4 w-px bg-line sm:block" />
          <span><b className="text-indigo">no search</b> · pure constraint math</span>
        </motion.div>
      </div>
    </section>
  );
}
