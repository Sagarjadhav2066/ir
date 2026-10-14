"""Disk-backed index construction for Lexis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

from .index import PositionalIndex, Posting
from .text import ParsedDocument, TextProcessor, parse_document


@dataclass(frozen=True)
class BuildMetrics:
    method: str
    documents: int
    blocks: int
    bytes_written: int
    peak_terms: int
    elapsed_seconds: float
    shards: int


def _document_postings(document: ParsedDocument) -> dict[str, dict[str, object]]:
    title_counts: dict[str, int] = {}
    body_counts: dict[str, int] = {}
    positions: dict[str, list[int]] = {}
    for position, token in enumerate(document.title_tokens):
        title_counts[token] = title_counts.get(token, 0) + 1
        positions.setdefault(token, []).append(position)
    offset = len(document.title_tokens)
    for position, token in enumerate(document.body_tokens, offset):
        body_counts[token] = body_counts.get(token, 0) + 1
        positions.setdefault(token, []).append(position)
    return {
        term: {
            "doc_id": document.doc_id,
            "positions": positions[term],
            "title_frequency": title_counts.get(term, 0),
            "body_frequency": body_counts.get(term, 0),
        }
        for term in positions
    }


def _write_block(block: dict[str, list[dict[str, object]]], path: Path) -> int:
    with path.open("w", encoding="utf-8") as handle:
        for term in sorted(block):
            handle.write(json.dumps([term, block[term]], separators=(",", ":")))
            handle.write("\n")
    return path.stat().st_size


def _merge_blocks(block_paths: list[Path], output_path: Path) -> int:
    merged: dict[str, list[dict[str, object]]] = {}
    for block_path in block_paths:
        with block_path.open(encoding="utf-8") as handle:
            for line in handle:
                term, postings = json.loads(line)
                merged.setdefault(term, []).extend(postings)
    with output_path.open("w", encoding="utf-8") as handle:
        for term in sorted(merged):
            postings = sorted(merged[term], key=lambda posting: posting["doc_id"])
            handle.write(json.dumps([term, postings], separators=(",", ":")))
            handle.write("\n")
    return output_path.stat().st_size


def _build_shard(
    paths: list[Path],
    shard_dir: Path,
    processor: TextProcessor,
    memory_limit_terms: int,
    method: str,
) -> tuple[list[Path], int, int, int]:
    block: dict[str, list[dict[str, object]]] = {}
    pair_block: list[tuple[str, dict[str, object]]] = []
    block_paths: list[Path] = []
    peak_terms = 0
    bytes_written = 0
    for path in paths:
        document = parse_document(
            path.stem, path.read_text(encoding="utf-8"), processor
        )
        document_postings = _document_postings(document)
        if method == "bsbi":
            pair_block.extend(document_postings.items())
            active_size = len(pair_block)
        else:
            for term, posting in document_postings.items():
                block.setdefault(term, []).append(posting)
            active_size = len(block)
        peak_terms = max(peak_terms, active_size)
        if active_size >= memory_limit_terms:
            if method == "bsbi":
                block = {}
                for term, posting in sorted(pair_block, key=lambda item: item[0]):
                    block.setdefault(term, []).append(posting)
            block_path = shard_dir / f"block_{len(block_paths):05d}.jsonl"
            bytes_written += _write_block(block, block_path)
            block_paths.append(block_path)
            block = {}
            pair_block = []
    if block or pair_block:
        if method == "bsbi":
            block = {}
            for term, posting in sorted(pair_block, key=lambda item: item[0]):
                block.setdefault(term, []).append(posting)
        block_path = shard_dir / f"block_{len(block_paths):05d}.jsonl"
        bytes_written += _write_block(block, block_path)
        block_paths.append(block_path)
    return block_paths, bytes_written, peak_terms, len(paths)


def build_disk_index(
    corpus_dir: Path,
    output_dir: Path,
    *,
    method: str = "spimi",
    memory_limit_terms: int = 10_000,
    shards: int = 1,
    processor: TextProcessor | None = None,
) -> tuple[PositionalIndex, BuildMetrics]:
    """Build and merge a disk-backed index, returning its queryable index."""
    if method not in {"spimi", "bsbi"}:
        raise ValueError("method must be 'spimi' or 'bsbi'")
    if memory_limit_terms < 1 or shards < 1:
        raise ValueError("memory_limit_terms and shards must be positive")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    active_processor = processor or TextProcessor()
    paths = sorted(corpus_dir.glob("*.txt"))
    shard_paths = [paths[index::shards] for index in range(shards)]
    started = time.perf_counter()
    shard_results: list[tuple[list[Path], int, int, int]] = []
    with ThreadPoolExecutor(max_workers=shards) as executor:
        futures = []
        for index, shard in enumerate(shard_paths):
            shard_dir = output_dir / f"shard_{index:02d}"
            shard_dir.mkdir()
            futures.append(
                executor.submit(
                    _build_shard,
                    shard,
                    shard_dir,
                    active_processor,
                    memory_limit_terms,
                    method,
                )
            )
        shard_results = [future.result() for future in futures]

    block_paths = [
        block_path
        for shard_blocks, _, _, _ in shard_results
        for block_path in shard_blocks
    ]
    merged_path = output_dir / "merged.jsonl"
    merge_bytes = _merge_blocks(block_paths, merged_path)
    elapsed = time.perf_counter() - started
    metrics = BuildMetrics(
        method=method,
        documents=len(paths),
        blocks=len(block_paths),
        bytes_written=sum(result[1] for result in shard_results) + merge_bytes,
        peak_terms=max((result[2] for result in shard_results), default=0),
        elapsed_seconds=elapsed,
        shards=shards,
    )
    postings: dict[str, tuple[Posting, ...]] = {}
    with merged_path.open(encoding="utf-8") as handle:
        for line in handle:
            term, raw_postings = json.loads(line)
            postings[term] = tuple(
                Posting(
                    doc_id=posting["doc_id"],
                    positions=tuple(posting["positions"]),
                    title_frequency=posting["title_frequency"],
                    body_frequency=posting["body_frequency"],
                )
                for posting in raw_postings
            )
    return PositionalIndex(postings, frozenset(path.stem for path in paths), active_processor), metrics


def save_metrics(metrics: BuildMetrics, path: Path) -> None:
    """Write metrics as JSON for reproducible report generation."""
    path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
