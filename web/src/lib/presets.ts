import type { DType } from "./hardware";
import type { TpuVersion } from "./hardware";
import type { OpType } from "./solver";

export interface Preset {
  id: string;
  label: string;
  hint: string;
  opType: OpType;
  M: number;
  N: number;
  K: number;
  tpu: TpuVersion;
  inputDtype: DType;
  outputDtype: DType;
  accumulatorDtype: DType;
}

const bf16: Pick<Preset, "inputDtype" | "outputDtype" | "accumulatorDtype"> = {
  inputDtype: "bfloat16",
  outputDtype: "bfloat16",
  accumulatorDtype: "float32",
};

export const PRESETS: Preset[] = [
  { id: "attn_qkv", label: "GPT-2 attn_qkv", hint: "fits with headroom", opType: "matmul", M: 512, N: 768, K: 768, tpu: "v5e", ...bf16 },
  { id: "ffn_up", label: "FFN up", hint: "wide N projection", opType: "matmul", M: 512, N: 3072, K: 768, tpu: "v5e", ...bf16 },
  { id: "ffn_down", label: "FFN down", hint: "deep K reduction · +5%", opType: "matmul", M: 512, N: 768, K: 3072, tpu: "v5e", ...bf16 },
  { id: "large", label: "Large 2048³", hint: "31% of v5e VMEM", opType: "matmul", M: 2048, N: 2048, K: 2048, tpu: "v5e", ...bf16 },
  { id: "rmsnorm", label: "RMSNorm", hint: "full-row tile · +3–4%", opType: "rmsnorm", M: 512, N: 768, K: 768, tpu: "v5e", ...bf16 },
  { id: "v4squeeze", label: "v4 squeeze 🔥", hint: "VMEM-bound → 75% full", opType: "matmul", M: 2048, N: 2048, K: 2048, tpu: "v4", ...bf16 },
];
