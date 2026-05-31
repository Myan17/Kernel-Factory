#!/usr/bin/env python
"""Production RAG corpus ingestion pipeline.

Clones the configured source repos, extracts semantic chunks from matching
Pallas kernel files, embeds them, and upserts into the LanceDB corpus.

Usage:
    uv run python scripts/ingest_rag.py
    uv run python scripts/ingest_rag.py --min-tier 1 --max-tier 1
    uv run python scripts/ingest_rag.py --no-clone
    uv run python scripts/ingest_rag.py --force
    uv run python scripts/ingest_rag.py --db-path /path/to/.lancedb
    uv run python scripts/ingest_rag.py --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from kernel_factory.chunker import KernelChunker
from kernel_factory.rag import ProductionRAG
from kernel_factory.rag_corpus import (
    CORPUS_SOURCES,
    EXCLUDE_PATTERNS,
    MIN_FILE_LINES,
    RepoSource,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "repos"
console = Console()


def _sanitize(name: str) -> str:
    return name.replace("/", "_")


def _clone_or_update(src: RepoSource, no_clone: bool) -> Path | None:
    dest = DATA_DIR / _sanitize(src.name)
    if no_clone:
        if not dest.exists():
            console.print(f"[yellow]--no-clone but {dest} missing; skipping {src.name}[/]")
            return None
        return dest

    from git import GitCommandError, Repo  # lazy import

    try:
        if dest.exists():
            Repo(dest).remotes.origin.pull()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            Repo.clone_from(src.url, dest, branch=src.branch, depth=1)
        return dest
    except GitCommandError as exc:
        console.print(f"[red]Clone/pull failed for {src.name}: {exc}[/]")
        return None
    except Exception as exc:  # network or git error
        console.print(f"[red]Error fetching {src.name}: {exc}[/]")
        return None


def _is_excluded(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in EXCLUDE_PATTERNS)


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(512)
    except OSError:
        return True


def _match_files(repo_dir: Path, src: RepoSource) -> list[Path]:
    matched: set[Path] = set()
    for glob in src.globs:
        for path in repo_dir.glob(glob):
            if not path.is_file():
                continue
            rel = str(path.relative_to(repo_dir))
            if _is_excluded(rel):
                continue
            if _is_binary(path):
                continue
            matched.add(path)
    return sorted(matched)


def _collect_chunks(repo_dir: Path, src: RepoSource, chunker: KernelChunker):
    files = _match_files(repo_dir, src)
    chunks = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text.splitlines()) < MIN_FILE_LINES:
            continue
        rel = str(path.relative_to(repo_dir))
        try:
            chunks += chunker.chunk_file(text, rel, src.name, tier=src.tier)
        except Exception as exc:  # defensive; chunker already guards SyntaxError
            console.print(f"[yellow]Skipping {rel}: {exc}[/]")
    return files, chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the production RAG corpus.")
    parser.add_argument("--db-path", default=str(REPO_ROOT / ".lancedb"))
    parser.add_argument("--min-tier", type=int, default=1)
    parser.add_argument("--max-tier", type=int, default=4)
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore chunk_id dedup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    console.rule("[bold]TPU Kernel Factory — RAG Corpus Ingestion[/]")

    sources = [
        s for s in CORPUS_SOURCES if args.min_tier <= s.tier <= args.max_tier
    ]
    if not sources:
        console.print("[red]No sources match the tier range.[/]")
        return 1

    chunker = KernelChunker()
    all_chunks = []

    if args.dry_run:
        console.print("[bold yellow]DRY RUN[/] — no clone, no DB writes.\n")
        table = Table(title="Would ingest")
        table.add_column("repo", style="cyan")
        table.add_column("tier", justify="right")
        table.add_column("globs", style="dim")
        for s in sources:
            table.add_row(s.name, str(s.tier), ", ".join(s.globs))
        console.print(table)
        console.print(f"\nTarget DB: {args.db_path}")
        console.print(f"Dedup: {'disabled (--force)' if args.force else 'enabled'}")
        return 0

    for src in sources:
        console.print(f"Fetching [cyan]{src.name}[/] (tier {src.tier})...")
        repo_dir = _clone_or_update(src, args.no_clone)
        if repo_dir is None:
            continue
        files, chunks = _collect_chunks(repo_dir, src, chunker)
        console.print(f"  {len(files)} files matched → {len(chunks)} chunks")
        all_chunks += chunks

    if not all_chunks:
        console.print("[red]No chunks collected. Nothing to ingest.[/]")
        return 1

    rag = ProductionRAG(db_path=Path(args.db_path))

    if args.force:
        # Drop the table so every chunk re-embeds.
        db = rag._connect()
        if rag.TABLE_NAME in db.list_tables().tables:
            db.drop_table(rag.TABLE_NAME)

    console.print(f"\nEmbedding + upserting [bold]{len(all_chunks)}[/] chunks...")
    with Progress(transient=True) as progress:
        task = progress.add_task("upsert", total=len(all_chunks))
        inserted = 0
        # Upsert in batches of 64 (embedding batch size).
        for i in range(0, len(all_chunks), 64):
            batch = all_chunks[i : i + 64]
            try:
                inserted += rag.upsert_chunks(batch)
            except Exception as exc:
                console.print(f"[red]Upsert failed: {exc}[/]")
                return 1
            progress.update(task, advance=len(batch))

    skipped = len(all_chunks) - inserted
    console.print(f"  {inserted} new · {skipped} skipped (duplicates)\n")

    stats = rag.stats()
    summary = Table(title="Corpus Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value")
    summary.add_row("Total chunks", str(stats["total_chunks"]))
    summary.add_row(
        "Kernel classes",
        ", ".join(f"{k}: {v}" for k, v in sorted(stats["by_kernel_class"].items())),
    )
    summary.add_row(
        "Source repos",
        ", ".join(f"{k}: {v}" for k, v in sorted(stats["by_source_repo"].items())),
    )
    summary.add_row(
        "Tiers",
        ", ".join(f"tier {k}: {v}" for k, v in sorted(stats["by_tier"].items())),
    )
    console.print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
