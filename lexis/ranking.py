"""TF-IDF ranking and evaluation for Lexis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import math
from pathlib import Path
import time

from .index import PositionalIndex


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float


@dataclass(frozen=True)
class RankingConfig:
    title_weight: float = 2.0
    body_weight: float = 1.0
    use_idf: bool = True
    length_normalization: bool = True
    raw_tf: bool = False


@dataclass(frozen=True)
class EvaluationMetrics:
    precision_at_10: float
    map: float
    ndcg_at_10: float


class RankedSearcher:
    def __init__(self, index: PositionalIndex, config: RankingConfig | None = None):
        self.index = index
        self.config = config or RankingConfig()
        self.document_norms = self._build_norms()

    def search(self, query: str, limit: int = 10) -> list[RankedDocument]:
        return self._search_terms(self.index.processor.tokenize(query), limit)

    def search_pruned(
        self, query: str, limit: int = 10, champion_size: int = 50
    ) -> tuple[list[RankedDocument], int, float]:
        started = time.perf_counter()
        terms = self.index.processor.tokenize(query)
        candidates: set[str] = set()
        for term in set(terms):
            postings = self.index.postings.get(term, ())
            ranked = sorted(
                postings,
                key=lambda posting: posting.term_frequency,
                reverse=True,
            )[:champion_size]
            candidates.update(posting.doc_id for posting in ranked)
        results = self._search_terms(terms, limit, candidates)
        return results, len(candidates), time.perf_counter() - started

    def _search_terms(
        self,
        terms: list[str],
        limit: int,
        candidates: set[str] | None = None,
    ) -> list[RankedDocument]:
        if limit < 1:
            return []
        unique_terms = set(terms).intersection(self.index.vocabulary)
        if not unique_terms:
            return []
        query_weights = {
            term: self._term_weight(term, terms.count(term), len(self.index.document_ids))
            for term in unique_terms
        }
        scores: dict[str, float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            for posting in self.index.postings[term]:
                if candidates is None or posting.doc_id in candidates:
                    scores[posting.doc_id] += query_weight * self._posting_weight(posting)
        ranked = []
        for doc_id, score in scores.items():
            if self.config.length_normalization:
                score /= self.document_norms.get(doc_id, 1.0)
            ranked.append(RankedDocument(doc_id, score))
        return sorted(ranked, key=lambda item: (-item.score, item.doc_id))[:limit]

    def _term_weight(self, term: str, frequency: int, documents: int) -> float:
        tf = 1.0 if self.config.raw_tf else 1.0 + math.log(frequency)
        if not self.config.use_idf:
            return tf
        df = len(self.index.postings.get(term, ()))
        return tf * math.log((documents + 1) / (df + 1)) + 1.0

    def _posting_weight(self, posting: object) -> float:
        title = posting.title_frequency * self.config.title_weight
        body = posting.body_frequency * self.config.body_weight
        tf = title + body
        return tf if self.config.raw_tf else (1.0 + math.log(tf) if tf else 0.0)

    def _build_norms(self) -> dict[str, float]:
        norms: dict[str, float] = defaultdict(float)
        for term, postings in self.index.postings.items():
            weight = self._term_weight(term, 1, len(self.index.document_ids))
            for posting in postings:
                value = weight * self._posting_weight(posting)
                norms[posting.doc_id] += value * value
        return {doc_id: math.sqrt(value) or 1.0 for doc_id, value in norms.items()}


def read_queries(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["query_id"]: row["query"] for row in csv.DictReader(handle, delimiter="\t")}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[row["query_id"]][row["doc_id"]] = int(row["grade"])
    return dict(qrels)


def evaluate(
    searcher: RankedSearcher,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> EvaluationMetrics:
    precisions: list[float] = []
    average_precisions: list[float] = []
    ndcgs: list[float] = []
    for query_id, query in queries.items():
        ranked = [item.doc_id for item in searcher.search(query, 10)]
        judgments = qrels.get(query_id, {})
        relevant = {doc_id for doc_id, grade in judgments.items() if grade > 0}
        precisions.append(sum(doc in relevant for doc in ranked) / 10)
        hits = 0
        total = 0.0
        for rank, doc_id in enumerate(ranked, 1):
            if doc_id in relevant:
                hits += 1
                total += hits / rank
        average_precisions.append(total / len(relevant) if relevant else 0.0)
        dcg = sum(
            (2 ** judgments.get(doc_id, 0) - 1) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked, 1)
        )
        ideal = sorted(judgments.values(), reverse=True)[:10]
        idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    count = max(1, len(queries))
    return EvaluationMetrics(sum(precisions) / count, sum(average_precisions) / count, sum(ndcgs) / count)
