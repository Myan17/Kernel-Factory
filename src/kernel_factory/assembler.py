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

        num_k_tiles = max(1, spec.K // config.block_k) if spec.op_type == "matmul" else 1

        return template.format(
            block_m=config.block_m,
            block_n=config.block_n,
            block_k=config.block_k,
            M=spec.M,
            N=spec.N,
            K=spec.K,
            input_dtype=config.input_dtype.value,
            output_dtype=config.output_dtype.value,
            accumulator_dtype=config.accumulator_dtype.value,
            num_k_tiles=num_k_tiles,
        )
