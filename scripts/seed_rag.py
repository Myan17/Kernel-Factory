#!/usr/bin/env python
"""One-time script to seed the LanceDB template store.

Usage:
    uv run python scripts/seed_rag.py
    uv run python scripts/seed_rag.py --force
    uv run python scripts/seed_rag.py --db-path /tmp/mydb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kernel_factory.rag import DIM, TemplateRAG, _template_label
from kernel_factory.templates import TEMPLATES


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the LanceDB template store.")
    parser.add_argument("--db-path", default=".lancedb", help="LanceDB directory")
    parser.add_argument("--force", action="store_true", help="Re-seed even if present")
    args = parser.parse_args()

    console = Console()
    rag = TemplateRAG(db_path=Path(args.db_path))

    try:
        inserted = rag.seed(force=args.force)
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Seed failed:[/] {exc}")
        return 1

    if inserted == 0:
        console.print(
            f"[yellow]Store at {args.db_path!r} already seeded[/] "
            "(use --force to re-seed)."
        )
        return 0

    table = Table(title=f"Seeded {inserted} template(s) -> {args.db_path}")
    table.add_column("op_type", style="cyan")
    table.add_column("template_name", style="green")
    table.add_column("vector_dim", justify="right", style="magenta")
    for op_type in TEMPLATES:
        table.add_row(op_type, _template_label(op_type), str(DIM))
    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
