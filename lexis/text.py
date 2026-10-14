"""Shared document and query text processing for Lexis."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


TOKEN_RE = re.compile(r"[a-z0-9]+")

# Kept configurable so the baseline can be reproduced with stop words disabled.
DEFAULT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class TextProcessor:
    """Normalize and tokenize text consistently across documents and queries."""

    remove_stop_words: bool = False
    stemming: bool = False
    stop_words: frozenset[str] = DEFAULT_STOP_WORDS

    def normalize(self, text: str) -> str:
        """Apply Unicode compatibility normalization and case folding."""
        return unicodedata.normalize("NFKC", text).casefold()

    def tokenize(self, text: str) -> list[str]:
        """Return normalized tokens in source order."""
        normalized = self.normalize(text)
        tokens = TOKEN_RE.findall(normalized)
        if self.stemming:
            tokens = [self._stem(token) for token in tokens]
        if self.remove_stop_words:
            return [token for token in tokens if token not in self.stop_words]
        return tokens

    @staticmethod
    def _stem(token: str) -> str:
        """Small deterministic suffix stemmer for controlled ablation runs."""
        for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
        return token


@dataclass(frozen=True)
class ParsedDocument:
    """A corpus document separated into title and body token streams."""

    doc_id: str
    title_tokens: tuple[str, ...]
    body_tokens: tuple[str, ...]

    @property
    def all_tokens(self) -> tuple[str, ...]:
        return self.title_tokens + self.body_tokens


def parse_document(
    doc_id: str, text: str, processor: TextProcessor | None = None
) -> ParsedDocument:
    """Parse the corpus format: title, blank line, then body."""
    active_processor = processor or TextProcessor()
    title, separator, body = text.partition("\n\n")
    if not separator:
        raise ValueError(f"{doc_id}: expected a title, blank line, and body")
    return ParsedDocument(
        doc_id=doc_id,
        title_tokens=tuple(active_processor.tokenize(title)),
        body_tokens=tuple(active_processor.tokenize(body)),
    )
