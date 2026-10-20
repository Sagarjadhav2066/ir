#!/usr/bin/env python3
"""Interactive query pipeline for the Lexis news archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.index import PositionalIndex
from lexis.ranking import RankedDocument, RankedSearcher
from lexis.text import TextProcessor


QUERY_TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|/\d+|[^\s()]+')
OPERATORS = frozenset({"AND", "OR", "NOT"})


@dataclass(frozen=True)
class SpellingSuggestion:
    original: str
    suggestions: tuple[str, ...]


def levenshtein(left: str, right: str, cutoff: int | None = None) -> int:
    """Return edit distance, stopping early when it exceeds cutoff."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        if cutoff is not None and min(current) > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


def suggestions_for_query(
    query: str,
    vocabulary: frozenset[str],
    processor: TextProcessor,
    limit: int = 3,
) -> tuple[SpellingSuggestion, ...]:
    """Find close vocabulary terms for absent, non-operator query words."""
    suggestions: list[SpellingSuggestion] = []
    seen: set[str] = set()
    for raw_token in QUERY_TOKEN_RE.findall(processor.normalize(query)):
        if raw_token in {"(", ")"} or raw_token.startswith("/"):
            continue
        if raw_token.startswith('"') and raw_token.endswith('"'):
            tokens = processor.tokenize(raw_token[1:-1])
        else:
            tokens = [raw_token.casefold()]
        for token in tokens:
            if token in OPERATORS or "*" in token or token in vocabulary or token in seen:
                continue
            seen.add(token)
            cutoff = max(1, min(3, len(token) // 3))
            candidates = [
                (levenshtein(token, term, cutoff), term)
                for term in vocabulary
                if abs(len(term) - len(token)) <= cutoff
            ]
            candidates = sorted(item for item in candidates if item[0] <= cutoff)
            suggestions.append(
                SpellingSuggestion(token, tuple(term for _, term in candidates[:limit]))
            )
    return tuple(suggestions)


def apply_suggestions(query: str, suggestions: tuple[SpellingSuggestion, ...]) -> str:
    """Replace misspelled standalone terms with their best suggestion."""
    corrected = query
    for item in suggestions:
        if item.suggestions:
            corrected = re.sub(
                rf"(?<![a-z0-9]){re.escape(item.original)}(?![a-z0-9])",
                item.suggestions[0],
                corrected,
                flags=re.IGNORECASE,
            )
    return corrected


def detect_mode(query: str) -> str:
    """Select Boolean retrieval for syntax-bearing queries, ranked retrieval otherwise."""
    if any(operator in QUERY_TOKEN_RE.findall(query.upper()) for operator in OPERATORS):
        return "boolean"
    if '"' in query or "(" in query or ")" in query or re.search(r"/\d+", query):
        return "boolean"
    if "*" in query:
        return "boolean"
    return "ranked"


def print_article(corpus: Path, doc_id: str, score: float | None = None, full: bool = False) -> None:
    path = corpus / f"{doc_id}.txt"
    text = path.read_text(encoding="utf-8")
    title, _, body = text.partition("\n\n")
    if score is None:
        print(f"{doc_id} | {title}")
    else:
        print(f"{doc_id} | score={score:.4f} | {title}")
    if full:
        print(body.strip())


def run_query(
    query: str,
    corpus: Path,
    index: PositionalIndex,
    searcher: RankedSearcher,
    mode: str,
    limit: int,
    full: bool,
) -> None:
    if not query.strip():
        print("Please enter a query.")
        return
    selected_mode = detect_mode(query) if mode == "auto" else mode
    suggestions = suggestions_for_query(query, index.vocabulary, index.processor)
    corrected = apply_suggestions(query, suggestions)
    for item in suggestions:
        if item.suggestions:
            print(f"Suggestion: {item.original} -> {', '.join(item.suggestions)}")
    if corrected != query:
        print(f"Searching corrected query: {corrected}")
    try:
        if selected_mode == "boolean":
            doc_ids = index.search(corrected)
            print(f"Mode: Boolean | Results: {len(doc_ids)}")
            for doc_id in doc_ids[:limit]:
                print_article(corpus, doc_id, full=full)
        else:
            results: list[RankedDocument] = searcher.search(corrected, limit)
            print(f"Mode: ranked free-text | Results shown: {len(results)}")
            for result in results:
                print_article(corpus, result.doc_id, result.score, full)
    except (ValueError, OSError) as error:
        print(f"Query error: {error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--query", help="Run one query and exit; omit for interactive mode.")
    parser.add_argument("--mode", choices=("auto", "boolean", "ranked"), default="auto")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Print full article bodies.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    index = PositionalIndex.from_corpus(args.corpus)
    searcher = RankedSearcher(index)
    if args.query is not None:
        run_query(args.query, args.corpus, index, searcher, args.mode, args.limit, args.full)
        return

    print("Lexis search ready. Enter a query, or type 'quit' to exit.")
    print("Examples: oil AND opec | \"oil prices\" | oil* | economy /5 growth")
    while True:
        try:
            query = input("\nlexis> ")
        except EOFError:
            print()
            break
        if query.strip().casefold() in {"quit", "exit"}:
            break
        run_query(query, args.corpus, index, searcher, args.mode, args.limit, args.full)


if __name__ == "__main__":
    main()
