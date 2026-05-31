"""FastMCP server exposing the kernel factory to IDE agents (e.g. Cursor).

Tools are defined as plain module-level functions (so they stay directly
callable and unit-testable) and registered with the MCP server below. Every
tool catches all exceptions and returns a JSON-serializable dict — never raises
to the MCP layer.
"""
from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from kernel_factory.assembler import Assembler
from kernel_factory.pipeline import KernelPipeline
from kernel_factory.rag import ProductionRAG
from kernel_factory.schemas import DType, HardwareLimits, LayerSpec
from kernel_factory.solver import TileSolver
from kernel_factory.templates import TEMPLATES

_HW = {
    "v4": HardwareLimits.for_v4,
    "v5e": HardwareLimits.for_v5e,
    "v6e": HardwareLimits.for_v6e,
}


def _hw(tpu_version: str) -> HardwareLimits:
    if tpu_version not in _HW:
        raise ValueError(f"Unknown tpu_version '{tpu_version}'")
    return _HW[tpu_version]()


def _spec(op_type, M, N, K, input_dtype, output_dtype, accum_dtype) -> LayerSpec:
    return LayerSpec(
        op_type=op_type, M=M, N=N, K=K,
        input_dtype=DType(input_dtype),
        output_dtype=DType(output_dtype),
        accumulator_dtype=DType(accum_dtype),
    )


def _config_dict(cfg) -> dict:
    return {
        "block_m": cfg.block_m,
        "block_n": cfg.block_n,
        "block_k": cfg.block_k,
        "stages": cfg.stages,
        "vmem_estimate_bytes": cfg.total_vmem_estimate_bytes,
        "vmem_utilization_pct": cfg.vmem_utilization_fraction * 100.0,
    }


def solve_tile_config(
    op_type: str,
    M: int,
    N: int,
    K: int,
    tpu_version: str = "v5e",
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
) -> dict:
    """
    Given matrix dimensions and TPU version, returns the solver-validated tile
    configuration. Use this BEFORE writing any Pallas kernel to get safe
    block_m, block_n, block_k values.

    Returns a dict with block_m/block_n/block_k/stages, vmem_estimate_bytes,
    vmem_utilization_pct, and tpu_version. On error returns {"error": ...}.
    """
    try:
        hw = _hw(tpu_version)
        spec = _spec(op_type, M, N, K, input_dtype, output_dtype, accum_dtype)
        cfg = TileSolver(hw).solve(spec)
        out = _config_dict(cfg)
        out["tpu_version"] = tpu_version
        return out
    except Exception as exc:
        return {"error": str(exc), "passed": False}


def retrieve_template(
    op_type: str,
    M: int,
    N: int,
    K: int,
    rag_path: str = ".lancedb",
) -> dict:
    """
    Retrieves the best-matching verified Pallas template for this op type and
    shape. The template contains {block_m}, {block_n}, {block_k} placeholders —
    do NOT fill them manually. Use assemble_kernel instead.

    Returns {"template_code", "source" ("rag"|"static_fallback"), "op_type"}.
    """
    try:
        spec = _spec(op_type, M, N, K, "bfloat16", "bfloat16", "float32")
        rag = ProductionRAG(db_path=Path(rag_path))
        if rag.is_seeded():
            try:
                code = rag.retrieve_one(spec)
            except Exception:
                code = None
            if code is not None:
                return {"template_code": code, "source": "rag", "op_type": op_type}
        code = TEMPLATES.get(op_type)
        if code is None:
            return {"error": f"No template for op_type '{op_type}'", "passed": False}
        return {"template_code": code, "source": "static_fallback", "op_type": op_type}
    except Exception as exc:
        return {"error": str(exc), "passed": False}


def assemble_kernel(
    op_type: str,
    M: int,
    N: int,
    K: int,
    tpu_version: str = "v5e",
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    rag_path: str = ".lancedb",
) -> dict:
    """
    Full solve + retrieve + assemble in one call. Returns a complete, runnable
    Pallas kernel string with all block sizes injected.

    This is the PRIMARY tool to use when asked to write a Pallas kernel. Never
    write Pallas kernel loops or memory movement logic from scratch — always use
    this tool and then make only parameter-level edits to the output.

    Returns {"assembled_code", "kernel_config", "template_source"}.
    """
    try:
        hw = _hw(tpu_version)
        spec = _spec(op_type, M, N, K, input_dtype, output_dtype, accum_dtype)
        cfg = TileSolver(hw).solve(spec)

        template = None
        template_source = "static_fallback"
        rag = ProductionRAG(db_path=Path(rag_path))
        if rag.is_seeded():
            try:
                template = rag.retrieve_one(spec)
                template_source = "rag"
            except Exception:
                template = None
        code = Assembler().assemble(spec, cfg, template=template)
        return {
            "assembled_code": code,
            "kernel_config": _config_dict(cfg),
            "template_source": template_source,
        }
    except Exception as exc:
        return {"error": str(exc), "passed": False}


def verify_kernel(
    op_type: str,
    M: int,
    N: int,
    K: int,
    tpu_version: str = "v5e",
    input_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    accum_dtype: str = "float32",
    db_path: str | None = None,
) -> dict:
    """
    Run the full pipeline and verification gate in CPU interpret mode.
    Returns pass/fail with numerical error and latency, plus kernel_config.

    Returns {"passed", "max_abs_error", "execution_latency_ms", "error_trace",
    "kernel_config"}. On setup error returns {"error": ..., "passed": False}.
    """
    try:
        hw = _hw(tpu_version)
        spec = _spec(op_type, M, N, K, input_dtype, output_dtype, accum_dtype)
        result = KernelPipeline(
            hw=hw, db_path=Path(db_path) if db_path else None
        ).run(spec)
        tr = result.test_result
        return {
            "passed": tr.passed,
            "max_abs_error": tr.max_abs_error,
            "execution_latency_ms": tr.execution_latency_ms,
            "error_trace": tr.error_trace,
            "kernel_config": _config_dict(result.kernel_config),
        }
    except Exception as exc:
        return {"error": str(exc), "passed": False}


mcp = FastMCP("kernel-factory")
mcp.tool(solve_tile_config)
mcp.tool(retrieve_template)
mcp.tool(assemble_kernel)
mcp.tool(verify_kernel)


if __name__ == "__main__":
    mcp.run(transport="stdio")
