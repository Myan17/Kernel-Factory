import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import {
  KG_EDGES,
  KG_NODES,
  CATEGORY_COLOR,
  type KGNode,
} from "../lib/graph";

const W = 940;
const H = 600;

interface Sim {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fx: number | null; // pinned (dragged) position
  fy: number | null;
}

function initialPositions(): Sim[] {
  // deterministic circular seed (avoid Math.random for a stable first paint)
  const n = KG_NODES.length;
  return KG_NODES.map((node, i) => {
    const a = (i / n) * Math.PI * 2;
    return {
      id: node.id,
      x: W / 2 + Math.cos(a) * 230,
      y: H / 2 + Math.sin(a) * 180,
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
    };
  });
}

export default function ForceGraph() {
  const reduce = useReducedMotion();
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<Sim[]>(initialPositions());
  const frameRef = useRef(0);
  const dragRef = useRef<{ id: string; moved: boolean } | null>(null);
  const [, setTick] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const degree = useMemo(() => {
    const d: Record<string, number> = {};
    for (const n of KG_NODES) d[n.id] = 0;
    for (const e of KG_EDGES) {
      d[e.source]++;
      d[e.target]++;
    }
    return d;
  }, []);

  const neighbors = useMemo(() => {
    const map: Record<string, Set<string>> = {};
    for (const n of KG_NODES) map[n.id] = new Set();
    for (const e of KG_EDGES) {
      map[e.source].add(e.target);
      map[e.target].add(e.source);
    }
    return map;
  }, []);

  // node radius: big enough for its label, nudged up by connection count
  const radius = useMemo(() => {
    const r: Record<string, number> = {};
    for (const n of KG_NODES) {
      const labelHalf = (n.label.length * 8 * 0.56) / 2;
      r[n.id] = Math.max(20 + degree[n.id] * 2.4, labelHalf + 7);
    }
    return r;
  }, [degree]);

  const step = () => {
    const sim = simRef.current;
    const REPULSE = 16000;
    const SPRING_LEN = 210;
    const SPRING_K = 0.012;
    const GRAVITY = 0.012;
    const DAMP = 0.9;
    const f = frameRef.current++;

    for (let i = 0; i < sim.length; i++) {
      const a = sim[i];
      if (a.fx !== null) continue; // pinned by drag
      let fx = 0;
      let fy = 0;
      for (let j = 0; j < sim.length; j++) {
        if (i === j) continue;
        const dx = a.x - sim[j].x;
        const dy = a.y - sim[j].y;
        const d2 = dx * dx + dy * dy + 0.01;
        const d = Math.sqrt(d2);
        const force = REPULSE / d2;
        fx += (dx / d) * force;
        fy += (dy / d) * force;
      }
      fx += (W / 2 - a.x) * GRAVITY;
      fy += (H / 2 - a.y) * GRAVITY;
      // perpetual gentle drift so the graph keeps floating
      fx += Math.sin(f * 0.012 + i * 1.7) * 0.9;
      fy += Math.cos(f * 0.015 + i * 2.3) * 0.9;
      a.vx = (a.vx + fx * 0.12) * DAMP;
      a.vy = (a.vy + fy * 0.12) * DAMP;
    }
    for (const e of KG_EDGES) {
      const a = sim.find((s) => s.id === e.source)!;
      const b = sim.find((s) => s.id === e.target)!;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const force = (d - SPRING_LEN) * SPRING_K;
      const fxv = (dx / d) * force;
      const fyv = (dy / d) * force;
      if (a.fx === null) { a.vx += fxv; a.vy += fyv; }
      if (b.fx === null) { b.vx -= fxv; b.vy -= fyv; }
    }
    for (const s of sim) {
      if (s.fx !== null && s.fy !== null) {
        s.x = s.fx;
        s.y = s.fy;
        continue;
      }
      const r = radius[s.id];
      s.x = Math.max(r + 6, Math.min(W - r - 6, s.x + s.vx));
      s.y = Math.max(r + 6, Math.min(H - r - 6, s.y + s.vy));
    }
  };

  useEffect(() => {
    if (reduce) {
      for (let i = 0; i < 320; i++) step();
      setTick((t) => t + 1);
      return;
    }
    let raf = 0;
    const loop = () => {
      step();
      setTick((t) => t + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduce]);

  // ── drag ──────────────────────────────────────────────────────────────────
  const toSvg = (clientX: number, clientY: number) => {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * W,
      y: ((clientY - rect.top) / rect.height) * H,
    };
  };

  const onPointerDownNode = (id: string) => (e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragRef.current = { id, moved: false };
    const node = simRef.current.find((s) => s.id === id)!;
    const p = toSvg(e.clientX, e.clientY);
    node.fx = p.x;
    node.fy = p.y;
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const drag = dragRef.current;
    if (!drag) return;
    drag.moved = true;
    const node = simRef.current.find((s) => s.id === drag.id)!;
    const p = toSvg(e.clientX, e.clientY);
    node.fx = p.x;
    node.fy = p.y;
    if (reduce) setTick((t) => t + 1);
  };
  const onPointerUp = (id: string) => () => {
    const drag = dragRef.current;
    const node = simRef.current.find((s) => s.id === id)!;
    node.fx = null;
    node.fy = null;
    if (drag && !drag.moved) {
      setSelected((cur) => (cur === id ? null : id)); // click = toggle highlight
    }
    dragRef.current = null;
  };

  const pos = (id: string) => simRef.current.find((s) => s.id === id)!;
  const node = (id: string): KGNode => KG_NODES.find((n) => n.id === id)!;

  const focus = selected ?? hover;
  const isLit = (id: string) => !focus || focus === id || neighbors[focus].has(id);
  const edgeLit = (s: string, t: string) => !focus || focus === s || focus === t;
  const detail = focus ? node(focus) : null;

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full touch-none select-none"
        style={{ minHeight: 360 }}
        role="img"
        aria-label="Knowledge graph of 11 node types and 10 edge types"
        onPointerMove={onPointerMove}
      >
        <defs>
          <marker id="kg-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L9,4.5 L0,9 Z" fill="rgb(var(--muted))" />
          </marker>
          <marker id="kg-arrow-lit" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L9,4.5 L0,9 Z" fill="rgb(var(--indigo))" />
          </marker>
        </defs>

        {KG_EDGES.map((e) => {
          const a = pos(e.source);
          const b = pos(e.target);
          const lit = edgeLit(e.source, e.target);
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const ux = dx / d;
          const uy = dy / d;
          const x1 = a.x + ux * (radius[e.source] + 2);
          const y1 = a.y + uy * (radius[e.source] + 2);
          const x2 = b.x - ux * (radius[e.target] + 8);
          const y2 = b.y - uy * (radius[e.target] + 8);
          const strong = focus && lit;
          return (
            <g key={e.id} opacity={lit ? 1 : 0.1}>
              <line
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={strong ? "rgb(var(--indigo))" : "rgb(var(--line))"}
                strokeWidth={strong ? 2.4 : 1.5}
                markerEnd={strong ? "url(#kg-arrow-lit)" : "url(#kg-arrow)"}
              />
              {strong && (
                <text
                  x={(x1 + x2) / 2}
                  y={(y1 + y2) / 2 - 4}
                  textAnchor="middle"
                  className="fill-indigo font-mono"
                  style={{ fontSize: 9, paintOrder: "stroke" }}
                  stroke="rgb(var(--surface))"
                  strokeWidth={3}
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
          const r = radius[n.id];
          const active = focus === n.id;
          return (
            <g
              key={n.id}
              opacity={lit ? 1 : 0.2}
              style={{ cursor: "grab" }}
              onPointerDown={onPointerDownNode(n.id)}
              onPointerUp={onPointerUp(n.id)}
              onPointerEnter={() => setHover(n.id)}
              onPointerLeave={() => setHover(null)}
            >
              <circle
                cx={p.x}
                cy={p.y}
                r={r}
                fill={`rgb(${color})`}
                stroke="rgb(var(--surface))"
                strokeWidth={active ? 4 : 2.5}
                style={active ? { filter: `drop-shadow(0 0 10px rgb(${color} / 0.8))` } : undefined}
              />
              <text
                x={p.x}
                y={p.y}
                textAnchor="middle"
                dominantBaseline="central"
                className="pointer-events-none fill-white font-mono font-semibold"
                style={{ fontSize: 8.5, paintOrder: "stroke" }}
                stroke="rgba(8,8,14,0.55)"
                strokeWidth={2.6}
              >
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* hover/selection detail */}
      <div className="pointer-events-none absolute bottom-3 left-3 max-w-xs rounded-xl border border-line bg-surface/90 px-3 py-2 backdrop-blur">
        {detail ? (
          <>
            <div className="font-mono text-sm font-bold text-ink">{detail.label}</div>
            <div className="text-xs text-muted">{detail.blurb}</div>
            <div className="mt-1 font-mono text-[10px] text-indigo">
              {degree[detail.id]} connection{degree[detail.id] === 1 ? "" : "s"}
            </div>
          </>
        ) : (
          <div className="text-xs text-muted">
            Click a node to pin its neighborhood · drag any node to explore →
          </div>
        )}
      </div>
    </div>
  );
}
