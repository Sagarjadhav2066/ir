"""Brute-force reference retrieval used as the correctness oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .text import ParsedDocument, TextProcessor, parse_document


TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|/\d+|[^\s()]+')
WILDCARD_RE = re.compile(r"^[a-z0-9*]+$")


@dataclass(frozen=True)
class TermNode:
    term: str


@dataclass(frozen=True)
class PhraseNode:
    terms: tuple[str, ...]


@dataclass(frozen=True)
class WildcardNode:
    pattern: str


@dataclass(frozen=True)
class ProximityNode:
    left: object
    distance: int
    right: object


@dataclass(frozen=True)
class NotNode:
    child: object


@dataclass(frozen=True)
class BinaryNode:
    operator: str
    left: object
    right: object


class QueryParser:
    """Recursive-descent parser with NOT > AND > OR precedence."""

    def __init__(self, query: str, processor: TextProcessor | None = None) -> None:
        self.processor = processor or TextProcessor()
        self.tokens = TOKEN_RE.findall(self.processor.normalize(query))
        self.position = 0

    def parse(self) -> object:
        if not self.tokens:
            raise ValueError("query is empty")
        expression = self._parse_or()
        if self.position != len(self.tokens):
            raise ValueError(f"unexpected token: {self.tokens[self.position]}")
        return expression

    def _parse_or(self) -> object:
        expression = self._parse_and()
        while self._accept("OR"):
            expression = BinaryNode("OR", expression, self._parse_and())
        return expression

    def _parse_and(self) -> object:
        expression = self._parse_not()
        while self._accept("AND"):
            expression = BinaryNode("AND", expression, self._parse_not())
        return expression

    def _parse_not(self) -> object:
        if self._accept("NOT"):
            return NotNode(self._parse_not())
        return self._parse_proximity()

    def _parse_proximity(self) -> object:
        expression = self._parse_primary()
        while self.position < len(self.tokens) and self.tokens[self.position].startswith("/"):
            distance = int(self.tokens[self.position][1:])
            self.position += 1
            expression = ProximityNode(
                expression, distance, self._parse_primary()
            )
        return expression

    def _parse_primary(self) -> object:
        if self._accept("("):
            expression = self._parse_or()
            self._expect(")")
            return expression
        if self.position >= len(self.tokens):
            raise ValueError("expected query term")
        token = self.tokens[self.position]
        self.position += 1
        if token.startswith('"') and token.endswith('"'):
            terms = tuple(self.processor.tokenize(token[1:-1]))
            if not terms:
                raise ValueError("quoted phrase is empty")
            return PhraseNode(terms)
        if "*" in token:
            if not WILDCARD_RE.fullmatch(token):
                raise ValueError(f"invalid wildcard term: {token}")
            return WildcardNode(token)
        return TermNode(token)

    def _accept(self, token: str) -> bool:
        if (
            self.position < len(self.tokens)
            and self.tokens[self.position].casefold() == token.casefold()
        ):
            self.position += 1
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            raise ValueError(f"expected {token}")


class BruteForceSearcher:
    """Evaluate parsed queries by scanning all normalized document tokens."""

    def __init__(
        self, documents: list[ParsedDocument], processor: TextProcessor | None = None
    ) -> None:
        self.processor = processor or TextProcessor()
        self.documents = {document.doc_id: document for document in documents}
        self.document_ids = frozenset(self.documents)
        self.vocabulary = frozenset(
            token
            for document in documents
            for token in document.all_tokens
        )

    @classmethod
    def from_corpus(
        cls, corpus_dir: Path, processor: TextProcessor | None = None
    ) -> "BruteForceSearcher":
        active_processor = processor or TextProcessor()
        paths = sorted(corpus_dir.glob("*.txt"))
        documents = [
            parse_document(
                path.stem, path.read_text(encoding="utf-8"), active_processor
            )
            for path in paths
        ]
        return cls(documents, active_processor)

    def search(self, query: str) -> list[str]:
        expression = QueryParser(query, self.processor).parse()
        return sorted(self._evaluate(expression))

    def _evaluate(self, expression: object) -> set[str]:
        if isinstance(expression, TermNode):
            return {
                doc_id
                for doc_id, document in self.documents.items()
                if expression.term in document.all_tokens
            }
        if isinstance(expression, PhraseNode):
            return {
                doc_id
                for doc_id, document in self.documents.items()
                if self._contains_phrase(document.all_tokens, expression.terms)
            }
        if isinstance(expression, WildcardNode):
            pattern = re.compile(
                "^" + re.escape(expression.pattern).replace(r"\*", ".*") + "$"
            )
            matching_terms = {
                term for term in self.vocabulary if pattern.fullmatch(term)
            }
            return {
                doc_id
                for doc_id, document in self.documents.items()
                if any(term in document.all_tokens for term in matching_terms)
            }
        if isinstance(expression, ProximityNode):
            return self._evaluate_proximity(expression)
        if isinstance(expression, NotNode):
            return set(self.document_ids).difference(self._evaluate(expression.child))
        if isinstance(expression, BinaryNode):
            left = self._evaluate(expression.left)
            right = self._evaluate(expression.right)
            if expression.operator == "AND":
                return left.intersection(right)
            return left.union(right)
        raise TypeError(f"unsupported expression: {expression!r}")

    def _evaluate_proximity(self, expression: ProximityNode) -> set[str]:
        left_terms = self._terms_for_position_check(expression.left)
        right_terms = self._terms_for_position_check(expression.right)
        return {
            doc_id
            for doc_id, document in self.documents.items()
            if self._within_distance(
                document.all_tokens, left_terms, right_terms, expression.distance
            )
        }

    def _terms_for_position_check(self, expression: object) -> set[str]:
        if isinstance(expression, TermNode):
            return {expression.term}
        if isinstance(expression, WildcardNode):
            pattern = re.compile(
                "^" + re.escape(expression.pattern).replace(r"\*", ".*") + "$"
            )
            return {term for term in self.vocabulary if pattern.fullmatch(term)}
        raise ValueError("proximity operands must be terms or wildcard terms")

    @staticmethod
    def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
        length = len(phrase)
        return any(
            tokens[index : index + length] == phrase
            for index in range(len(tokens) - length + 1)
        )

    @staticmethod
    def _within_distance(
        tokens: tuple[str, ...],
        left_terms: set[str],
        right_terms: set[str],
        distance: int,
    ) -> bool:
        left_positions = [
            index
            for index, token in enumerate(tokens)
            if token in left_terms
        ]
        right_positions = [
            index
            for index, token in enumerate(tokens)
            if token in right_terms
        ]
        return any(
            abs(right - left) <= distance
            for left in left_positions
            for right in right_positions
        )
