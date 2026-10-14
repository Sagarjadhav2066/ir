"""In-memory positional inverted index for Lexis retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from .reference import (
    BinaryNode,
    NotNode,
    PhraseNode,
    ProximityNode,
    QueryParser,
    TermNode,
    WildcardNode,
)
from .text import ParsedDocument, TextProcessor, parse_document


@dataclass(frozen=True)
class Posting:
    """A term's occurrence information for one document."""

    doc_id: str
    positions: tuple[int, ...]
    title_frequency: int
    body_frequency: int

    @property
    def term_frequency(self) -> int:
        return self.title_frequency + self.body_frequency


class PositionalIndex:
    """Positional index with title/body frequencies and active-document state."""

    def __init__(
        self,
        postings: dict[str, tuple[Posting, ...]],
        document_ids: frozenset[str],
        processor: TextProcessor,
    ) -> None:
        self.postings = postings
        self.document_ids = document_ids
        self.processor = processor
        self.vocabulary = frozenset(postings)
        self._wildcard_cache: dict[str, tuple[str, ...]] = {}

    def add_document(self, document: ParsedDocument) -> None:
        """Add or replace one document without rebuilding other postings."""
        self.delete_document(document.doc_id)
        mutable = defaultdict(list)
        for term, posting in _postings_for_document(document).items():
            mutable[term].append(posting)
        postings = dict(self.postings)
        for term, values in mutable.items():
            postings[term] = tuple(sorted((*postings.get(term, ()), *values), key=lambda item: item.doc_id))
        self.postings = postings
        self.document_ids = frozenset((*self.document_ids, document.doc_id))
        self.vocabulary = frozenset(postings)
        self._wildcard_cache.clear()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document logically from all postings."""
        self.document_ids = frozenset(item for item in self.document_ids if item != doc_id)
        self.postings = {
            term: tuple(posting for posting in postings if posting.doc_id != doc_id)
            for term, postings in self.postings.items()
            if any(posting.doc_id != doc_id for posting in postings)
        }
        self.vocabulary = frozenset(self.postings)
        self._wildcard_cache.clear()

    @classmethod
    def build(
        cls, documents: list[ParsedDocument], processor: TextProcessor | None = None
    ) -> "PositionalIndex":
        active_processor = processor or TextProcessor()
        mutable: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        document_ids = frozenset(document.doc_id for document in documents)
        for document in documents:
            title_counts: dict[str, int] = defaultdict(int)
            body_counts: dict[str, int] = defaultdict(int)
            positions: dict[str, list[int]] = defaultdict(list)
            for position, token in enumerate(document.title_tokens):
                title_counts[token] += 1
                positions[token].append(position)
            offset = len(document.title_tokens)
            for position, token in enumerate(document.body_tokens, offset):
                body_counts[token] += 1
                positions[token].append(position)
            for term in positions:
                mutable[term][document.doc_id] = {
                    "positions": tuple(positions[term]),
                    "title_frequency": title_counts[term],
                    "body_frequency": body_counts[term],
                }

        postings = {
            term: tuple(
                Posting(doc_id, **values)
                for doc_id, values in sorted(documents_for_term.items())
            )
            for term, documents_for_term in mutable.items()
        }
        return cls(postings, document_ids, active_processor)

    @classmethod
    def from_corpus(
        cls, corpus_dir: Path, processor: TextProcessor | None = None
    ) -> "PositionalIndex":
        active_processor = processor or TextProcessor()
        documents = [
            parse_document(
                path.stem,
                path.read_text(encoding="utf-8"),
                active_processor,
            )
            for path in sorted(corpus_dir.glob("*.txt"))
        ]
        return cls.build(documents, active_processor)

    def search(self, query: str) -> list[str]:
        expression = QueryParser(query, self.processor).parse()
        return sorted(self._evaluate(expression))

    def _evaluate(self, expression: object) -> set[str]:
        if isinstance(expression, TermNode):
            return {
                posting.doc_id
                for posting in self.postings.get(expression.term, ())
                if posting.doc_id in self.document_ids
            }
        if isinstance(expression, PhraseNode):
            return {
                posting.doc_id
                for posting in self._phrase_postings(expression.terms)
                if posting.doc_id in self.document_ids
            }
        if isinstance(expression, WildcardNode):
            return set().union(
                *(self._evaluate(TermNode(term)) for term in self._matching_terms(expression.pattern))
            )
        if isinstance(expression, ProximityNode):
            return self._evaluate_proximity(expression)
        if isinstance(expression, NotNode):
            return set(self.document_ids).difference(self._evaluate(expression.child))
        if isinstance(expression, BinaryNode):
            left = self._evaluate(expression.left)
            right = self._evaluate(expression.right)
            return left.intersection(right) if expression.operator == "AND" else left.union(right)
        raise TypeError(f"unsupported expression: {expression!r}")

    def _matching_terms(self, pattern: str) -> tuple[str, ...]:
        if pattern not in self._wildcard_cache:
            regex = re.compile(
                "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
            )
            self._wildcard_cache[pattern] = tuple(
                term for term in sorted(self.vocabulary) if regex.fullmatch(term)
            )
        return self._wildcard_cache[pattern]

    def _phrase_postings(self, terms: tuple[str, ...]) -> tuple[Posting, ...]:
        if not terms:
            return ()
        candidate_ids = set(self._posting_ids(terms[0]))
        for term in terms[1:]:
            candidate_ids.intersection_update(self._posting_ids(term))
        matches: list[Posting] = []
        for doc_id in sorted(candidate_ids):
            positions = self._positions(terms[0], doc_id)
            for offset, term in enumerate(terms[1:], 1):
                positions = {
                    position
                    for position in positions
                    if position + offset in self._positions(term, doc_id)
                }
            if positions:
                matches.append(self._posting(terms[0], doc_id))
        return tuple(matches)

    def _evaluate_proximity(self, expression: ProximityNode) -> set[str]:
        left_terms = self._position_terms(expression.left)
        right_terms = self._position_terms(expression.right)
        candidate_ids = set().union(
            *(self._posting_ids(term) for term in left_terms)
        ).intersection(
            set().union(*(self._posting_ids(term) for term in right_terms))
        )
        return {
            doc_id
            for doc_id in candidate_ids
            if any(
                abs(left - right) <= expression.distance
                for left in self._all_positions(left_terms, doc_id)
                for right in self._all_positions(right_terms, doc_id)
            )
        }

    def _position_terms(self, expression: object) -> tuple[str, ...]:
        if isinstance(expression, TermNode):
            return (expression.term,)
        if isinstance(expression, WildcardNode):
            return self._matching_terms(expression.pattern)
        raise ValueError("proximity operands must be terms or wildcard terms")

    def _posting_ids(self, term: str) -> set[str]:
        return {posting.doc_id for posting in self.postings.get(term, ())}

    def _posting(self, term: str, doc_id: str) -> Posting:
        return next(posting for posting in self.postings[term] if posting.doc_id == doc_id)

    def _positions(self, term: str, doc_id: str) -> set[int]:
        for posting in self.postings.get(term, ()):
            if posting.doc_id == doc_id:
                return set(posting.positions)
        return set()

    def _all_positions(self, terms: tuple[str, ...], doc_id: str) -> set[int]:
        return set().union(*(self._positions(term, doc_id) for term in terms))


def _postings_for_document(document: ParsedDocument) -> dict[str, Posting]:
    title_counts: dict[str, int] = defaultdict(int)
    body_counts: dict[str, int] = defaultdict(int)
    positions: dict[str, list[int]] = defaultdict(list)
    for position, token in enumerate(document.title_tokens):
        title_counts[token] += 1
        positions[token].append(position)
    offset = len(document.title_tokens)
    for position, token in enumerate(document.body_tokens, offset):
        body_counts[token] += 1
        positions[token].append(position)
    return {
        term: Posting(
            document.doc_id,
            tuple(term_positions),
            title_counts[term],
            body_counts[term],
        )
        for term, term_positions in positions.items()
    }
