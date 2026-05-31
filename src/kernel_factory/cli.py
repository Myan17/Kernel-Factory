"""User-facing Typer CLI: run, inspect, seed."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kernel_factory.assembler import Assembler
from kernel_factory.pipeline import KernelPipeline
from kernel_factory.rag import ProductionRAG, TemplateRAG
from kernel_factory.schemas import DType, HardwareLimits, LayerSpec
from kernel_factory.solver import TileSolver

app = typer.Typer(help="TPU Kernel Factory — solve, assemble, and verify Pallas kernels.")
console = Console()

_HW = {
    "v4": HardwareLimits.for_v4,
    "v5e": HardwareLimits.for_v5e,
    "v6e": HardwareLimits.for_v6e,
}
_SUPPORTED_OPS = {"matmul", "rmsnorm"}


def _validate(op: str, tpu: str) -> HardwareLimits:
    if op not in _SUPPORTED_OPS:
        console.print(
            f"[bold red]Error:[/] unsupported op '{op}'. "
            f"Choose one of: {', '.join(sorted(_SUPPORTED_OPS))}"
        )
        raise typer.Exit(1)
    if tpu not in _HW:
        console.print(
            f"[bold red]Error:[/] unknown TPU '{tpu}'. "
            f"Choose one of: {', '.join(_HW)}"
        )
        raise typer.Exit(1)
    return _HW[tpu]()


def _spec(op, M, N, K, input_dtype, output_dtype, accum_dtype) -> LayerSpec:
    return LayerSpec(
        op_type=op, M=M, N=N, K=K,
        input_dtype=DType(input_dtype),
        output_dtype=DType(output_dtype),
        accumulator_dtype=DType(accum_dtype),
    )


@app.command()
def run(
    op: str = typer.Option(..., "--op", help="matmul | rmsnorm"),
    M: int = typer.Option(..., "--M"),
    N: int = typer.Option(..., "--N"),
    K: int = typer.Option(..., "--K"),
    tpu: str = typer.Option("v5e", "--tpu", help="v4 | v5e | v6e"),
    input_dtype: str = typer.Option("bfloat16", "--input-dtype"),
    output_dtype: str = typer.Option("bfloat16", "--output-dtype"),
    accum_dtype: str = typer.Option("float32", "--accum-dtype"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="SQLite results path"),
    rag_path: Path = typer.Option(Path(".lancedb"), "--rag-path", help="LanceDB corpus path"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", help="Write kernel code here"),
):
    """Solve + assemble + verify, then print a result table."""
    hw = _validate(op, tpu)
    spec = _spec(op, M, N, K, input_dtype, output_dtype, accum_dtype)

    # Use the production corpus if it has been ingested; otherwise fall back to
    # static templates inside the pipeline.
    rag = None
    candidate = ProductionRAG(db_path=rag_path)
    if candidate.is_seeded():
        rag = candidate

    try:
        result = KernelPipeline(hw=hw, rag=rag, db_path=db_path).run(spec)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Pipeline error:[/] {exc}")
        raise typer.Exit(1)

    cfg = result.kernel_config
    tr = result.test_result
    vmem_mib = cfg.total_vmem_estimate_bytes / (1024 * 1024)
    budget_pct = cfg.total_vmem_estimate_bytes / hw.vmem_budget_bytes * 100
    if tr.passed:
        verdict = f"[green]✅ PASSED[/] (max_err={tr.max_abs_error:.2e})"
    else:
        verdict = "[red]❌ FAILED[/]"

    table = Table(title=f"Kernel Factory — {op}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Op type", op)
    table.add_row("TPU version", tpu)
    table.add_row("block_m / n / k", f"{cfg.block_m} / {cfg.block_n} / {cfg.block_k}")
    table.add_row("VMEM estimate", f"{vmem_mib:.2f} MiB ({budget_pct:.1f}% of budget)")
    table.add_row("Template source", result.template_source)
    table.add_row("Verification", verdict)
    if tr.execution_latency_ms is not None:
        table.add_row("Latency", f"{tr.execution_latency_ms:.1f} ms")
    console.print(table)

    if output_file is not None:
        output_file.write_text(result.assembled_code)
        console.print(f"[dim]Wrote assembled kernel to {output_file}[/]")


@app.command()
def inspect(
    op: str = typer.Option(..., "--op", help="matmul | rmsnorm"),
    M: int = typer.Option(..., "--M"),
    N: int = typer.Option(..., "--N"),
    K: int = typer.Option(..., "--K"),
    tpu: str = typer.Option("v5e", "--tpu"),
    input_dtype: str = typer.Option("bfloat16", "--input-dtype"),
    output_dtype: str = typer.Option("bfloat16", "--output-dtype"),
    accum_dtype: str = typer.Option("float32", "--accum-dtype"),
):
    """Dry run: show the solver's tile choice and an assembled-code preview (no verify)."""
    hw = _validate(op, tpu)
    spec = _spec(op, M, N, K, input_dtype, output_dtype, accum_dtype)

    try:
        cfg = TileSolver(hw).solve(spec)
        code = Assembler().assemble(spec, cfg)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Solver error:[/] {exc}")
        raise typer.Exit(1)

    vmem_mib = cfg.total_vmem_estimate_bytes / (1024 * 1024)
    table = Table(title=f"Solver dry run — {op} on {tpu}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("block_m", str(cfg.block_m))
    table.add_row("block_n", str(cfg.block_n))
    table.add_row("block_k", str(cfg.block_k))
    table.add_row("stages", str(cfg.stages))
    table.add_row("VMEM estimate", f"{vmem_mib:.2f} MiB")
    table.add_row("VMEM utilization", f"{cfg.vmem_utilization_fraction * 100:.2f}%")
    console.print(table)

    console.print("[bold]Assembled code (first 30 lines):[/]")
    preview = "\n".join(code.splitlines()[:30])
    console.print(preview)


@app.command()
def seed(
    rag_path: Path = typer.Option(Path(".lancedb"), "--rag-path", help="LanceDB store path"),
    force: bool = typer.Option(False, "--force", help="Re-seed even if present"),
):
    """Seed the LanceDB RAG store with the verified templates."""
    rag = TemplateRAG(db_path=rag_path)
    inserted = rag.seed(force=force)
    if inserted == 0:
        console.print(
            f"[yellow]Store at {rag_path} already seeded[/] (use --force to re-seed)."
        )
        return
    table = Table(title=f"Seeded {inserted} template(s) -> {rag_path}")
    table.add_column("Status", style="green")
    table.add_row(f"{inserted} templates embedded and stored")
    console.print(table)


if __name__ == "__main__":
    app()
