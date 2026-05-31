import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { SolveResult } from "../lib/solver";

// Stage indices shared by the Live Demo and the Architecture diagram so they
// light up together.  0 Solve · 1 Retrieve · 2 Assemble · 3 Verify
export const STAGES = ["Solve", "Retrieve", "Assemble", "Verify"] as const;
export type StageName = (typeof STAGES)[number];

export interface VerifyState {
  status: "pass" | "fail";
  maxAbsError: number;
  latencyMs: number;
}

export interface PipelineState {
  activeStage: number; // -1 = idle
  reached: number; // highest stage reached this run (-1 idle)
  running: boolean;
  result: SolveResult | null;
  verify: VerifyState | null;
  runToken: number; // increments each run, to trigger pulses
}

interface PipelineCtx extends PipelineState {
  set: (patch: Partial<PipelineState>) => void;
  reset: () => void;
}

const IDLE: PipelineState = {
  activeStage: -1,
  reached: -1,
  running: false,
  result: null,
  verify: null,
  runToken: 0,
};

const Ctx = createContext<PipelineCtx | null>(null);

export function PipelineProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PipelineState>(IDLE);

  const value = useMemo<PipelineCtx>(
    () => ({
      ...state,
      set: (patch) => setState((s) => ({ ...s, ...patch })),
      reset: () => setState((s) => ({ ...IDLE, runToken: s.runToken })),
    }),
    [state]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function usePipeline(): PipelineCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("usePipeline must be used inside PipelineProvider");
  return ctx;
}
