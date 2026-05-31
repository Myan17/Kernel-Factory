import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import {
  KG_EDGES,
  KG_NODES,
  CATEGORY_COLOR,
  type KGNode,
} from "../lib/graph";

const W = 820;
const H = 540;

interface Sim {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

function initialPositions(): Sim[] {
  // deterministic circular seed (no Math.random for stable SSR/first paint)
  const n = KG_NODES.length;
  return KG_NODES.map((node, i) => {
    const a = (i / n) * Math.PI * 2;
    return {
      id: node.id,
      x: W / 2 + Math.cos(a) * 200,
      y: H / 2 + Math.sin(a) * 170,
      vx: 0,
      vy: 0,
    };
  });
}

function step(sim: Sim[], iterations = 1) {
  const REPULSE = 6200;
  const SPRING_LEN = 155;
  const SPRING_K = 0.022;
  const GRAVITY = 0.02;
  const DAMP = 0.85;

  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < sim.length; i++) {
      let fx = 0;
      let fy = 0;
      for (let j = 0; j < sim.length; j++) {
        if (i === j) continue;
        const dx = sim[i].x - sim[j].x;
        const dy = sim[i].y - sim[j].y;
        const d2 = dx * dx + dy * dy + 0.01;
        const d = Math.sqrt(d2);
        const f = REPULSE / d2;
        fx += (dx / d) * f;
        fy += (dy / d) * f;
      }
      fx += (W / 2 - sim[i].x) * GRAVITY;
      fy += (H / 2 - sim[i].y) * GRAVITY;
      sim[i].vx = (sim[i].vx + fx) * DAMP;
      sim[i].vy = (sim[i].vy + fy) * DAMP;
    }
    for (const e of KG_EDGES) {
      const a = sim.find((s) => s.id === e.source)!;
      const b = sim.find((s) => s.id === e.target)!;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const f = (d - SPRING_LEN) * SPRING_K;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
    for (const s of sim) {
      s.x = Math.max(60, Math.min(W - 60, s.x + s.vx));
      s.y = Math.max(40, Math.min(H - 40, s.y + s.vy));
    }
  }
}

export default function ForceGraph() {
  const reduce = useReducedMotion();
  const simRef = useRef<Sim[]>(initialPositions());
  const [, setTick] = useState(0);
  const [hover, setHover] = useState<string | null>(null);

  const neighbors = useMemo(() => {
    const map: Record<string, Set<string>> = {};
    for (const n of KG_NODES) map[n.id] = new Set();
    for (const e of KG_EDGES) {
      map[e.source].add(e.target);
      map[e.target].add(e.source);
    }
    return map;
  }, []);

  useEffect(() => {
    if (reduce) {
      step(simRef.current, 280); // settle once, render static
      setTick((t) => t + 1);
      return;
    }
    let raf = 0;
    let frames = 0;
    const loop = () => {
      step(simRef.current, 1);
      setTick((t) => t + 1);
      frames++;
      // keep a gentle idle motion but back off after settling
      if (frames < 600) raf = requestAnimationFrame(loop);
      else raf = window.setTimeout(() => requestAnimationFrame(loop), 90) as unknown as number;
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(raf);
    };
  }, [reduce]);

  const pos = (id: string) => simRef.current.find((s) => s.id === id)!;
  const node = (id: string): KGNode => KG_NODES.find((n) => n.id === id)!;

  const isLit = (id: string) =>
    !hover || hover === id || neighbors[hover].has(id);
  const edgeLit = (s: string, t: string) =>
    !hover || hover === s || hover === t;

  const hovered = hover ? node(hover) : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none"
        role="img"
        aria-label="Knowledge graph of 11 node types and 10 edge types"
      >
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="rgb(var(--muted))" />
          </marker>
        </defs>

        {KG_EDGES.map((e) => {
          const a = pos(e.source);
          const b = pos(e.target);
          const lit = edgeLit(e.source, e.target);
          const mx = (a.x + b.x) / 2;
          const my = (a.y + b.y) / 2;
          return (
            <g key={e.id} opacity={lit ? 1 : 0.12}>
              <line
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={hover && lit ? "rgb(var(--indigo))" : "rgb(var(--line))"}
                strokeWidth={hover && lit ? 2 : 1.4}
                markerEnd="url(#arrow)"
              />
              {hover && lit && (
                <text
                  x={mx}
                  y={my - 4}
                  textAnchor="middle"
                  className="fill-muted font-mono"
                  style={{ fontSize: 9 }}
                >
                  {e.label}
                </text>
              )}
            </g>
          );
        })}

        {KG_NODES.map((n) => {
          const p = pos(n.id);
          const lit = isLit(n.id);
          const color = CATEGORY_COLOR[n.category];
          const r = hover === n.id ? 11 : 8;
          return (
            <g
              key={n.id}
              opacity={lit ? 1 : 0.22}
              onMouseEnter={() => setHover(n.id)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: "pointer" }}
            >
              <circle cx={p.x} cy={p.y} r={r} fill={`rgb(${color})`} stroke="rgb(var(--surface))" strokeWidth={2} />
              <text
                x={p.x}
                y={p.y - 14}
                textAnchor="middle"
                className="fill-ink font-mono font-medium"
                style={{ fontSize: 11 }}
              >
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* hover detail */}
      <div className="pointer-events-none absolute bottom-3 left-3 max-w-xs rounded-xl border border-line bg-surface/90 px-3 py-2 backdrop-blur">
        {hovered ? (
          <>
            <div className="font-mono text-sm font-bold text-ink">{hovered.label}</div>
            <div className="text-xs text-muted">{hovered.blurb}</div>
          </>
        ) : (
          <div className="text-xs text-muted">Hover a node to trace its provenance edges →</div>
        )}
      </div>
    </div>
  );
}
