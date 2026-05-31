import Reveal from "../components/Reveal";
import SectionHeading from "../components/SectionHeading";

const CARDS = [
  {
    icon: "🧨",
    tone: "coral",
    title: "LLMs hallucinate kernels",
    body: "Let a model free-write Pallas/Mosaic and it invents primitives, mismatches shapes, and crashes the compiler. Plausible code, broken kernel.",
  },
  {
    icon: "🐌",
    tone: "duck",
    title: "Autotuning brute-forces",
    body: "Searching 10³–10⁵ tile configs is slow, and it's blind to cold shapes you haven't profiled yet. You pay the search every time.",
  },
  {
    icon: "📐",
    tone: "mint",
    title: "We compute the tile",
    body: "From the TPU's VMEM, vector and sublane widths we solve the largest valid tile directly — instant, deterministic, and numerically verifiable.",
  },
] as const;

const TONE: Record<string, string> = {
  coral: "from-coral/20 to-coral/5 text-coral",
  duck: "from-duck/25 to-duck/5 text-duck",
  mint: "from-mint/20 to-mint/5 text-mint",
};

export default function Problem() {
  return (
    <section id="problem" className="scroll-mt-20 py-20 sm:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="The wedge"
          title={<>Two ways to get a kernel are <span className="text-gradient">broken</span>. There's a third.</>}
          subtitle="Generation hallucinates. Search is slow. The factory replaces both with hardware math."
        />

        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {CARDS.map((c, i) => (
            <Reveal key={c.title} delay={i * 0.1}>
              <div className="group h-full rounded-xl2 border border-line bg-surface p-6 shadow-soft transition hover:-translate-y-1 hover:shadow-lift">
                <div className={`mb-4 grid h-12 w-12 place-items-center rounded-xl bg-gradient-to-br text-2xl ${TONE[c.tone]}`}>
                  {c.icon}
                </div>
                <h3 className="font-display text-xl font-bold">{c.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{c.body}</p>
                {i === 2 && (
                  <div className="mt-4 inline-flex rounded-lg bg-mint/15 px-2.5 py-1 font-mono text-xs font-semibold text-mint ring-1 ring-mint/40">
                    this project →
                  </div>
                )}
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
