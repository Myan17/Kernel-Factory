import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { tokenizePython, TOKEN_CLASS } from "../lib/highlight";

interface Props {
  code: string;
  /** "show" renders {placeholders} as chips; "fill" renders the integer values. */
  placeholderMode?: "show" | "fill" | "none";
  subs?: Record<string, string>;
  filename?: string;
  /** copy text override; defaults to the resolved code */
  copyText?: string;
  className?: string;
  maxHeight?: number;
}

export default function CodeBlock({
  code,
  placeholderMode = "none",
  subs = {},
  filename,
  copyText,
  className = "",
  maxHeight = 460,
}: Props) {
  const tokens = useMemo(() => tokenizePython(code), [code]);
  const [copied, setCopied] = useState(false);

  const resolved = useMemo(() => {
    if (placeholderMode === "fill") {
      return code.replace(/\{(\w+)\}/g, (m, k: string) => (k in subs ? subs[k] : m));
    }
    return code;
  }, [code, placeholderMode, subs]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(copyText ?? resolved);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div
      className={`group relative overflow-hidden rounded-2xl border border-line bg-ink/[0.97] shadow-soft ${className}`}
    >
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
        <span className="h-3 w-3 rounded-full bg-coral/80" />
        <span className="h-3 w-3 rounded-full bg-duck/80" />
        <span className="h-3 w-3 rounded-full bg-mint/80" />
        <span className="ml-2 font-mono text-xs text-white/55">
          {filename ?? "generated_kernel.py"}
        </span>
        <button
          onClick={onCopy}
          className="ml-auto rounded-lg border border-white/15 px-2.5 py-1 font-mono text-[11px] text-white/70 transition hover:border-mint/60 hover:text-mint"
        >
          {copied ? "copied ✓" : "copy"}
        </button>
      </div>
      <pre
        className="code-scroll overflow-auto p-4 font-mono text-[12.5px] leading-relaxed text-white/90"
        style={{ maxHeight }}
      >
        <code>
          {tokens.map((t, i) => {
            if (t.type === "placeholder" && placeholderMode !== "none") {
              const key = t.key!;
              const value = key in subs ? subs[key] : key;
              if (placeholderMode === "show") {
                return (
                  <span
                    key={i}
                    className="rounded-[5px] bg-duck/25 px-1 text-duck ring-1 ring-duck/50"
                  >
                    {t.text}
                  </span>
                );
              }
              // fill — morph the placeholder into the integer/value
              return (
                <AnimatePresence mode="popLayout" key={i}>
                  <motion.span
                    key={value}
                    initial={{ backgroundColor: "rgba(61,220,151,0.45)" }}
                    animate={{ backgroundColor: "rgba(61,220,151,0)" }}
                    transition={{ duration: 1.1 }}
                    className="rounded-[5px] px-0.5 font-medium text-mint"
                  >
                    {value}
                  </motion.span>
                </AnimatePresence>
              );
            }
            return (
              <span key={i} className={TOKEN_CLASS[t.type]}>
                {t.text}
              </span>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
