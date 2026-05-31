from __future__ import annotations
from kernel_factory.schemas import KernelConfig, LayerSpec
from kernel_factory.templates import TEMPLATES


class Assembler:
    def assemble(
        self,
        spec: LayerSpec,
        config: KernelConfig,
        template: str | None = None,
    ) -> str:
        # When a template string is supplied (e.g. retrieved from RAG), use it
        # directly. Otherwise fall back to the static verified template registry.
        if template is None:
            template = TEMPLATES.get(spec.op_type)
        if template is None:
            raise ValueError(f"No template registered for op_type='{spec.op_type}'")

        if spec.op_type == "flash_attention":
            num_k_tiles = max(1, (spec.seq_len or spec.K) // config.block_k)
        elif spec.op_type == "matmul":
            num_k_tiles = max(1, spec.K // config.block_k)
        elif spec.op_type == "fused_matmul_rmsnorm":
            num_k_tiles = max(1, spec.K // config.block_k)
        else:
            num_k_tiles = 1

        head_dim = spec.head_dim or spec.K
        scale = round(head_dim ** -0.5, 6)

        # Targeted placeholder substitution rather than str.format(): RAG-retrieved
        # templates may be real kernel code containing literal braces (dicts,
        # f-strings) that would make .format() raise. Replacing only our known
        # placeholders is identical for the verified templates and safe otherwise.
        replacements = {
            "{block_m}": str(config.block_m),
            "{block_n}": str(config.block_n),
            "{block_k}": str(config.block_k),
            "{M}": str(spec.M),
            "{N}": str(spec.N),
            "{K}": str(spec.K),
            "{input_dtype}": config.input_dtype.value,
            "{output_dtype}": config.output_dtype.value,
            "{accumulator_dtype}": config.accumulator_dtype.value,
            "{num_k_tiles}": str(num_k_tiles),
            "{scale}": str(scale),
        }
        out = template
        for placeholder, value in replacements.items():
            out = out.replace(placeholder, value)
        return out
