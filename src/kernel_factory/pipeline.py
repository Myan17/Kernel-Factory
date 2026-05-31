"""End-to-end orchestrator: LayerSpec + HardwareLimits -> solve -> retrieve(RAG)
-> assemble -> verify -> log(KG). Returns a PipelineResult."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from kernel_factory.assembler import Assembler
from kernel_factory.schemas import HardwareLimits, KernelConfig, LayerSpec, TestResult
from kernel_factory.solver import TileSolver
from kernel_factory.templates import TEMPLATES
from kernel_factory.verify import VerificationGate, VerifyMode

log = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    layer_spec: LayerSpec
    kernel_config: KernelConfig
    assembled_code: str
    test_result: TestResult
    template_source: str  # "rag" | "static_fallback"
    kg_logged: bool


class KernelPipeline:
    def __init__(
        self,
        hw: HardwareLimits,
        rag: "TemplateRAG | None" = None,  # noqa: F821 - forward ref, avoid import cycle
        kg: "KernelFactoryKG | None" = None,  # noqa: F821
        db_path: Optional[Path] = None,
    ):
        self.hw = hw
        self.rag = rag
        self.kg = kg
        self.db_path = db_path

    def run(self, spec: LayerSpec) -> PipelineResult:
        # 1. Solve — raises ValueError for unsupported op_type, before RAG is touched.
        config = TileSolver(self.hw).solve(spec)

        # 2. Retrieve a template (RAG-first, static fallback).
        template, template_source = self._resolve_template(spec)

        # 3. Assemble runnable kernel code from the resolved template.
        assembled_code = Assembler().assemble(spec, config, template=template)

        # 4. Verify in CPU interpret mode (real VerificationGate API).
        test_result = VerificationGate(
            spec, config, mode=VerifyMode.CPU_INTERPRET, db_path=self.db_path
        ).run()

        # 5. Best-effort KG log.
        kg_logged = self._log_to_kg(spec, config, assembled_code)

        return PipelineResult(
            layer_spec=spec,
            kernel_config=config,
            assembled_code=assembled_code,
            test_result=test_result,
            template_source=template_source,
            kg_logged=kg_logged,
        )

    def _resolve_template(self, spec: LayerSpec) -> tuple[str | None, str]:
        if self.rag is not None:
            try:
                return self.rag.retrieve(spec), "rag"
            except Exception as exc:  # RAG not seeded / unavailable
                log.warning("RAG retrieve failed (%s); using static template", exc)
        return TEMPLATES.get(spec.op_type), "static_fallback"

    def _log_to_kg(
        self, spec: LayerSpec, config: KernelConfig, code: str
    ) -> bool:
        if self.kg is None:
            return False
        try:
            kernel_id = self._kernel_id(spec, config)
            created_at = datetime.now(timezone.utc).isoformat()
            self.kg.upsert_generated_kernel(kernel_id, code, created_at)
            return True
        except Exception as exc:  # Kuzu unavailable / write failed
            log.warning("KG log failed (%s); continuing", exc)
            return False

    @staticmethod
    def _kernel_id(spec: LayerSpec, config: KernelConfig) -> str:
        key = (
            f"{spec.op_type}_{spec.M}x{spec.N}x{spec.K}_"
            f"{config.block_m}x{config.block_n}x{config.block_k}"
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        return f"{spec.op_type}_{digest}"
