import { useEffect, useRef } from "react";
import { useReducedMotion } from "framer-motion";

// Flowing tiles that drift around a grid — evokes matrix tiling. Canvas-based,
// cheap, and it pauses when off-screen / reduced-motion.
const COLORS = ["91,91,214", "139,92,246", "61,220,151", "255,197,61"];

export default function TileField() {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let w = 0;
    let h = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    type Tile = { x: number; y: number; s: number; vx: number; vy: number; c: string; a: number };
    let tiles: Tile[] = [];

    const seed = () => {
      const count = Math.min(46, Math.floor((w * h) / 26000));
      tiles = Array.from({ length: count }, () => {
        const s = 14 + Math.random() * 46;
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          s,
          vx: (Math.random() - 0.5) * 0.22,
          vy: (Math.random() - 0.5) * 0.22,
          c: COLORS[Math.floor(Math.random() * COLORS.length)],
          a: 0.05 + Math.random() * 0.12,
        };
      });
    };

    const resize = () => {
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      seed();
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      for (const t of tiles) {
        if (!reduce) {
          t.x += t.vx;
          t.y += t.vy;
          if (t.x < -t.s) t.x = w + t.s;
          if (t.x > w + t.s) t.x = -t.s;
          if (t.y < -t.s) t.y = h + t.s;
          if (t.y > h + t.s) t.y = -t.s;
        }
        const r = 6;
        const x = t.x;
        const y = t.y;
        const s = t.s;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + s, y, x + s, y + s, r);
        ctx.arcTo(x + s, y + s, x, y + s, r);
        ctx.arcTo(x, y + s, x, y, r);
        ctx.arcTo(x, y, x + s, y, r);
        ctx.closePath();
        ctx.fillStyle = `rgba(${t.c},${t.a})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(${t.c},${t.a + 0.12})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (!reduce) raf = requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [reduce]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden />;
}
