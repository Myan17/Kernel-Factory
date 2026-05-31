import { useEffect, useState } from "react";
import { useTheme } from "../state/theme";

const LINKS = [
  { href: "#problem", label: "Problem" },
  { href: "#demo", label: "Live demo" },
  { href: "#architecture", label: "Pipeline" },
  { href: "#proof", label: "Proof" },
  { href: "#graph", label: "Graph" },
  { href: "#tooling", label: "Tooling" },
];

const REPO = "https://github.com/Myan17/QuackHacks";

export default function Nav() {
  const { theme, toggle } = useTheme();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all ${
        scrolled ? "glass border-b border-line shadow-soft" : ""
      }`}
    >
      <nav className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-2 font-display text-lg font-extrabold tracking-tight">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-indigo to-violet text-base shadow-glow">
            🦆
          </span>
          <span>Kernel<span className="text-gradient">Factory</span></span>
        </a>

        <div className="ml-auto hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="rounded-lg px-3 py-1.5 text-sm font-medium text-muted transition hover:bg-indigo/10 hover:text-ink"
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-2 md:ml-2">
          <button
            onClick={toggle}
            aria-label="Toggle dark mode"
            className="grid h-9 w-9 place-items-center rounded-lg border border-line text-ink transition hover:bg-indigo/10"
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="hidden rounded-lg bg-ink px-3.5 py-2 text-sm font-semibold text-canvas transition hover:opacity-90 sm:block"
          >
            GitHub ↗
          </a>
        </div>
      </nav>
    </header>
  );
}
