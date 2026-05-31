import type { TpuVersion } from "./hardware";

// Public on-demand list-price ESTIMATES (USD per chip-hour), order-of-magnitude
// from Google Cloud TPU pricing. Used only for the cost calculator — labelled as
// estimates in the UI.
export const TPU_PRICE_PER_HR: Record<TpuVersion, number> = {
  v4: 3.22,
  v5e: 1.2,
  v6e: 2.7,
};

export function fmtUSD(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1000) return `$${Math.round(n).toLocaleString("en-US")}`;
  if (abs >= 1) return `$${n.toFixed(2)}`;
  return `$${n.toFixed(4)}`;
}
