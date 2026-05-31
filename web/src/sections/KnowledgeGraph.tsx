import Reveal from "../components/Reveal";
import SectionHeading from "../components/SectionHeading";
import ForceGraph from "../components/ForceGraph";
import { CATEGORY_COLOR, CATEGORY_LABEL, type NodeCategory } from "../lib/graph";

const CATS: NodeCategory[] = ["spec", "hardware", "kernel", "result", "knowledge"];

export default function KnowledgeGraph() {
  return (
    <section id="graph" className="scroll-mt-20 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="Provenance"
          title={<>Every kernel has a <span className="text-gradient">paper trail</span></>}
          subtitle="A Kuzu knowledge graph — 11 node types, 10 edge types — records where each spec, tile, kernel, compile, benchmark and failure came from. Hover to trace the neighborhood."
        />

        <Reveal delay={0.1}>
          <div className="mt-10 overflow-hidden rounded-xl2 border border-line bg-surface shadow-soft">
            <div className="flex flex-wrap gap-3 border-b border-line px-5 py-3">
              {CATS.map((c) => (
                <span key={c} className="inline-flex items-center gap-1.5 font-mono text-xs text-muted">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: `rgb(${CATEGORY_COLOR[c]})` }} />
                  {CATEGORY_LABEL[c]}
                </span>
              ))}
              <span className="ml-auto font-mono text-xs text-muted">11 nodes · 10 edges</span>
            </div>
            <div className="p-2 sm:p-4">
              <ForceGraph />
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
