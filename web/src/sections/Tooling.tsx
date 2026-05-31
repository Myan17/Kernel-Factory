import { useState } from "react";
import Reveal from "../components/Reveal";
import SectionHeading from "../components/SectionHeading";
import Terminal, { type Line } from "../components/Terminal";

const MCP_TOOLS = [
  { name: "solve_tile_config", purpose: "Compute the optimal tile from (M, N, K) + TPU limits — pure constraint math." },
  { name: "assemble_kernel", purpose: "Primary tool: solve → retrieve → inject → return runnable Pallas source.", primary: true },
  { name: "retrieve_template", purpose: "Fetch the verified Pallas template for an op from the LanceDB corpus." },
  { name: "verify_kernel", purpose: "Run in CPU interpret-mode and check numerically vs a pure-JAX baseline." },
  { name: "search_corpus", purpose: "Semantic search over the ingested JAX / Pallas reference corpus." },
];

const CLI = [
  "kernel-factory run     --op matmul --M 1024 --N 1024 --K 512 --tpu v5e",
  "kernel-factory inspect --op rmsnorm --M 512 --N 4096 --K 4096 --tpu v5e",
  "kernel-factory seed    --rag-path .lancedb",
];

const TERMINAL: Line[] = [
  { text: "kernel-factory run --op matmul --M 1024 --N 1024 --K 512 --tpu v5e", kind: "cmd" },
  { text: "", kind: "dim" },
  { text: "▸ solve     TileSolver(v5e)  budget=96.00 MiB", kind: "dim" },
  { text: "  block_m=1024  block_n=1024  block_k=512", kind: "head" },
  { text: "  vmem_estimate=8.00 MiB  util=6.25%  num_k_tiles=1", kind: "out" },
  { text: "▸ retrieve  template=matmul.pallas  (verified)", kind: "dim" },
  { text: "▸ assemble  injected 10 placeholders → matmul_kernel.py", kind: "dim" },
  { text: "▸ verify    interpret-mode vs jnp baseline (atol/rtol 1e-2)", kind: "dim" },
  { text: "  max_abs_error=3.7e-03   compiled=True", kind: "out" },
  { text: "", kind: "dim" },
  { text: "✅ PASS  kernel written to out/matmul_1024x1024x512_v5e.py", kind: "ok" },
  { text: "   provenance → Kuzu  ·  result → SQLite", kind: "dim" },
];

export default function Tooling() {
  return (
    <section id="tooling" className="scroll-mt-20 bg-surface/40 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="Integrate"
          title={<>Ships as a <span className="text-gradient">CLI</span> and an <span className="text-gradient">MCP server</span></>}
          subtitle="Five FastMCP tools so IDE agents like Cursor can call the factory directly — or drive it from the Typer CLI."
        />

        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          {/* MCP tools */}
          <div className="space-y-3">
            <div className="font-mono text-xs uppercase tracking-wider text-muted">FastMCP · 5 tools</div>
            {MCP_TOOLS.map((t, i) => (
              <Reveal key={t.name} delay={i * 0.06}>
                <div
                  className={`rounded-xl2 border bg-surface p-4 shadow-soft transition hover:-translate-y-0.5 hover:shadow-lift ${
                    t.primary ? "border-indigo/50 ring-1 ring-indigo/20" : "border-line"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <code className="font-mono text-sm font-bold text-indigo">{t.name}</code>
                    {t.primary && (
                      <span className="rounded-md bg-indigo/15 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-indigo">
                        primary
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-muted">{t.purpose}</p>
                </div>
              </Reveal>
            ))}
          </div>

          {/* CLI + terminal */}
          <div className="space-y-4">
            <div className="font-mono text-xs uppercase tracking-wider text-muted">Typer CLI</div>
            <CliBlock />
            <Terminal lines={TERMINAL} />
          </div>
        </div>
      </div>
    </section>
  );
}

function CliBlock() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(CLI.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* noop */
    }
  };
  return (
    <div className="relative overflow-hidden rounded-xl2 border border-line bg-ink/[0.97] p-4 shadow-soft">
      <button
        onClick={copy}
        className="absolute right-3 top-3 rounded-lg border border-white/15 px-2.5 py-1 font-mono text-[11px] text-white/70 transition hover:border-mint/60 hover:text-mint"
      >
        {copied ? "copied ✓" : "copy"}
      </button>
      <pre className="code-scroll overflow-auto font-mono text-[12px] leading-relaxed text-white/85">
        {CLI.map((c, i) => (
          <div key={i}>
            <span className="text-mint">$ </span>
            {c}
          </div>
        ))}
      </pre>
    </div>
  );
}
