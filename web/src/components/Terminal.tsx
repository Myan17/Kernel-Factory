import { useEffect, useRef, useState } from "react";
import { useInView, useReducedMotion } from "framer-motion";

export interface Line {
  text: string;
  kind?: "cmd" | "out" | "ok" | "dim" | "head";
}

const KIND_CLASS: Record<string, string> = {
  cmd: "text-white",
  out: "text-white/80",
  ok: "text-mint",
  dim: "text-white/45",
  head: "text-duck",
};

export default function Terminal({ lines, title = "zsh — kernel-factory" }: { lines: Line[]; title?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);
  const [typed, setTyped] = useState("");

  // type the first line (the command), then reveal output lines one by one
  useEffect(() => {
    if (!inView) return;
    if (reduce) {
      setShown(lines.length);
      setTyped(lines[0]?.text ?? "");
      return;
    }
    let i = 0;
    let timer: number;
    const cmd = lines[0]?.text ?? "";
    const typeCmd = () => {
      if (i <= cmd.length) {
        setTyped(cmd.slice(0, i));
        i++;
        timer = window.setTimeout(typeCmd, 28);
      } else {
        setShown(1);
        revealOut(1);
      }
    };
    const revealOut = (n: number) => {
      if (n > lines.length) return;
      setShown(n + 1);
      timer = window.setTimeout(() => revealOut(n + 1), 95);
    };
    typeCmd();
    return () => clearTimeout(timer);
  }, [inView, reduce, lines]);

  return (
    <div ref={ref} className="overflow-hidden rounded-xl2 border border-line bg-ink/[0.97] shadow-soft">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
        <span className="h-3 w-3 rounded-full bg-coral/80" />
        <span className="h-3 w-3 rounded-full bg-duck/80" />
        <span className="h-3 w-3 rounded-full bg-mint/80" />
        <span className="ml-2 font-mono text-xs text-white/55">{title}</span>
      </div>
      <div className="code-scroll overflow-auto p-4 font-mono text-[12.5px] leading-relaxed" style={{ minHeight: 260 }}>
        {lines.map((l, idx) => {
          if (idx >= shown) return null;
          const isCmd = l.kind === "cmd" || idx === 0;
          const content = idx === 0 ? typed : l.text;
          return (
            <div key={idx} className={`whitespace-pre ${KIND_CLASS[l.kind ?? (isCmd ? "cmd" : "out")]}`}>
              {isCmd && <span className="text-mint">❯ </span>}
              {content}
              {idx === 0 && typed.length < (lines[0]?.text.length ?? 0) && (
                <span className="ml-0.5 inline-block w-[7px] animate-pulse bg-white align-middle" style={{ height: "1em" }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
