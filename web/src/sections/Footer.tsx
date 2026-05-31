const STACK = ["JAX / Pallas", "Pydantic", "Kuzu", "LanceDB", "FastMCP", "Typer", "SQLite"];
const REPO = "https://github.com/Myan17/QuackHacks";

export default function Footer() {
  return (
    <footer className="border-t border-line py-14">
      <div className="mx-auto max-w-7xl px-4 sm:px-6">
        <div className="flex flex-col items-start justify-between gap-8 md:flex-row md:items-center">
          <div>
            <div className="flex items-center gap-2 font-display text-xl font-extrabold">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-indigo to-violet text-base shadow-glow">
                🦆
              </span>
              Kernel<span className="text-gradient">Factory</span>
            </div>
            <p className="mt-3 max-w-md text-sm text-muted">
              The same kind of model that hallucinates broken kernels, turned into a machine that can
              only emit correct ones — because the numbers come from hardware math, not a guess.
            </p>
          </div>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="rounded-xl bg-ink px-5 py-3 text-sm font-semibold text-canvas transition hover:opacity-90"
          >
            GitHub ↗
          </a>
        </div>

        <div className="mt-10 flex flex-wrap gap-2">
          {STACK.map((s) => (
            <span
              key={s}
              className="rounded-lg border border-line bg-surface px-3 py-1.5 font-mono text-xs text-muted"
            >
              {s}
            </span>
          ))}
        </div>

        <div className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-line pt-6 font-mono text-xs text-muted sm:flex-row sm:items-center">
          <span>🦆 Built at QuackHacks · supported ops: matmul · rmsnorm · TPUs: v4 / v5e / v6e</span>
          <span>computed, not guessed</span>
        </div>
      </div>
    </footer>
  );
}
