#!/usr/bin/env python3
"""Validate the Lexis assignment data files and corpus layout."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


DOC_ID_RE = re.compile(r"doc_\d{5}\.txt$")
QUERY_ID_RE = re.compile(r"(?:q|b)\d{3}$")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
    return reader.fieldnames, rows


def require_columns(path: Path, actual: list[str], expected: set[str]) -> None:
    missing = expected.difference(actual)
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")


def validate_corpus(root: Path) -> tuple[int, int]:
    corpus = root / "corpus"
    files = sorted(corpus.glob("*.txt"))
    invalid_names = [file.name for file in files if not DOC_ID_RE.fullmatch(file.name)]
    if invalid_names:
        raise ValueError(f"corpus: invalid document names: {invalid_names[:3]}")

    empty = []
    malformed = []
    total_bytes = 0
    for file in files:
        text = file.read_text(encoding="utf-8")
        total_bytes += file.stat().st_size
        if not text.strip():
            empty.append(file.name)
        lines = text.splitlines()
        if len(lines) < 3 or lines[1].strip():
            malformed.append(file.name)

    if empty:
        raise ValueError(f"corpus: empty documents: {empty[:3]}")
    if malformed:
        raise ValueError(
            "corpus: expected title, blank line, and body in: "
            f"{malformed[:3]}"
        )
    return len(files), total_bytes


def validate_queries(root: Path, filename: str, expected_prefix: str) -> int:
    path = root / filename
    fields, rows = read_tsv(path)
    require_columns(path, fields, {"query_id", "query"})
    seen: set[str] = set()
    for row in rows:
        query_id = row["query_id"]
        if query_id in seen:
            raise ValueError(f"{path}: duplicate query ID: {query_id}")
        if not query_id.startswith(expected_prefix) or not QUERY_ID_RE.fullmatch(
            query_id
        ):
            raise ValueError(f"{path}: invalid query ID: {query_id}")
        if not row["query"].strip():
            raise ValueError(f"{path}: empty query: {query_id}")
        seen.add(query_id)
    return len(rows)


def validate_qrels(root: Path, document_ids: set[str], query_ids: set[str]) -> int:
    path = root / "qrels.tsv"
    fields, rows = read_tsv(path)
    require_columns(path, fields, {"query_id", "doc_id", "grade"})
    for row in rows:
        if row["query_id"] not in query_ids:
            raise ValueError(f"{path}: unknown query ID: {row['query_id']}")
        if row["doc_id"] not in document_ids:
            raise ValueError(f"{path}: unknown document ID: {row['doc_id']}")
        try:
            grade = int(row["grade"])
        except ValueError as error:
            raise ValueError(f"{path}: non-integer grade: {row['grade']}") from error
        if grade not in {0, 1, 2}:
            raise ValueError(f"{path}: invalid grade: {grade}")
    return len(rows)


def validate_boolean_tests(root: Path, document_ids: set[str]) -> int:
    path = root / "boolean_tests.tsv"
    fields, rows = read_tsv(path)
    require_columns(
        path, fields, {"query_id", "query", "type", "num_results", "expected_doc_ids"}
    )
    for row in rows:
        if not row["query_id"].startswith("b"):
            raise ValueError(f"{path}: invalid test ID: {row['query_id']}")
        expected = (
            [] if not row["expected_doc_ids"].strip() else row["expected_doc_ids"].split(",")
        )
        if len(expected) != int(row["num_results"]):
            raise ValueError(
                f"{path}: result count mismatch for {row['query_id']}: "
                f"declared {row['num_results']}, listed {len(expected)}"
            )
        unknown = set(expected).difference(document_ids)
        if unknown:
            raise ValueError(
                f"{path}: unknown documents for {row['query_id']}: {sorted(unknown)[:3]}"
            )
    return len(rows)


def validate_misspellings(root: Path) -> int:
    path = root / "misspellings.tsv"
    fields, rows = read_tsv(path)
    require_columns(path, fields, {"misspelled", "accepted_corrections"})
    for row in rows:
        if not row["misspelled"].strip() or not row["accepted_corrections"].strip():
            raise ValueError(f"{path}: empty spelling pair")
    return len(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    document_count, corpus_bytes = validate_corpus(root)
    document_ids = {f"doc_{number:05d}" for number in range(1, document_count + 1)}
    actual_document_ids = {
        path.stem for path in (root / "corpus").glob("*.txt")
    }
    if actual_document_ids != document_ids:
        missing = sorted(document_ids - actual_document_ids)
        extra = sorted(actual_document_ids - document_ids)
        raise ValueError(f"corpus: missing={missing[:3]}, extra={extra[:3]}")

    train_count = validate_queries(root, "queries_train.tsv", "q")
    test_count = validate_queries(root, "queries_test.tsv", "q")
    train_fields, train_rows = read_tsv(root / "queries_train.tsv")
    test_fields, test_rows = read_tsv(root / "queries_test.tsv")
    query_ids = {row["query_id"] for row in train_rows + test_rows}
    qrels_count = validate_qrels(root, document_ids, query_ids)
    boolean_count = validate_boolean_tests(root, document_ids)
    misspelling_count = validate_misspellings(root)

    summary = {
        "documents": document_count,
        "corpus_bytes": corpus_bytes,
        "training_queries": train_count,
        "test_queries": test_count,
        "qrels": qrels_count,
        "boolean_tests": boolean_count,
        "misspellings": misspelling_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
