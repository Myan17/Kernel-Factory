import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { usePipeline } from "../state/pipeline";

const QUACKS = ["Quack! ✅", "Verified!", "It runs! 🟢", "Numbers, not guesses."];

export default function Duck() {
  const { verify, runToken } = usePipeline();
  const reduce = useReducedMotion();
  const [say, setSay] = useState<string | null>(null);
  const [bounce, setBounce] = useState(0);

  useEffect(() => {
    if (verify?.status === "pass") {
      setSay(QUACKS[runToken % QUACKS.length]);
      setBounce((b) => b + 1);
      const t = setTimeout(() => setSay(null), 2600);
      return () => clearTimeout(t);
    }
  }, [verify, runToken]);

  return (
    <div className="pointer-events-none fixed bottom-5 right-5 z-40 hidden select-none sm:block">
      <AnimatePresence>
        {say && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.9 }}
            className="mb-2 ml-auto w-max max-w-[180px] rounded-2xl rounded-br-sm border border-line bg-surface px-3 py-1.5 font-mono text-xs text-ink shadow-lift"
          >
            {say}
          </motion.div>
        )}
      </AnimatePresence>
      <motion.div
        key={bounce}
        animate={
          reduce
            ? {}
            : say
            ? { y: [0, -22, 0, -10, 0], rotate: [0, -8, 8, -4, 0] }
            : { y: [0, -5, 0] }
        }
        transition={
          say
            ? { duration: 0.9, ease: "easeOut" }
            : { duration: 3.4, repeat: Infinity, ease: "easeInOut" }
        }
        className="ml-auto w-max text-5xl drop-shadow-[0_10px_20px_rgba(91,91,214,0.35)]"
        aria-hidden
      >
        🦆
      </motion.div>
    </div>
  );
}
