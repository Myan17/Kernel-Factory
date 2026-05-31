import { lazy, Suspense } from "react";
import { ThemeProvider } from "./state/theme";
import { PipelineProvider } from "./state/pipeline";
import Nav from "./components/Nav";
import Duck from "./components/Duck";
import Hero from "./sections/Hero";
import Problem from "./sections/Problem";
import LiveDemo from "./sections/LiveDemo";
import Architecture from "./sections/Architecture";
import KnowledgeGraph from "./sections/KnowledgeGraph";
import Tooling from "./sections/Tooling";
import Footer from "./sections/Footer";

// Proof pulls in Recharts — split it into its own chunk so it doesn't bloat
// the initial load (the hero must animate instantly).
const Proof = lazy(() => import("./sections/Proof"));

export default function App() {
  return (
    <ThemeProvider>
      <PipelineProvider>
        <Nav />
        <main>
          <Hero />
          <Problem />
          <LiveDemo />
          <Architecture />
          <Suspense fallback={<div id="proof" className="scroll-mt-20 py-28 text-center font-mono text-sm text-muted">loading charts…</div>}>
            <Proof />
          </Suspense>
          <KnowledgeGraph />
          <Tooling />
        </main>
        <Footer />
        <Duck />
      </PipelineProvider>
    </ThemeProvider>
  );
}
