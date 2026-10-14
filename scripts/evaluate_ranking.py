#!/usr/bin/env python3
"""Run Lexis ranking evaluation and champion-list pruning measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.index import PositionalIndex
from lexis.ranking import RankedSearcher, evaluate, read_qrels, read_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--qrels", type=Path, default=Path("qrels.tsv"))
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--champion-size", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = PositionalIndex.from_corpus(args.corpus)
    searcher = RankedSearcher(index)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    exhaustive = evaluate(searcher, queries, qrels)
    overlaps = []
    for query_id, query in queries.items():
        exact = {item.doc_id for item in searcher.search(query)}
        pruned, scored, latency = searcher.search_pruned(
            query, champion_size=args.champion_size
        )
        overlap = len(exact.intersection(item.doc_id for item in pruned)) / 10
        overlaps.append(
            {
                "query_id": query_id,
                "overlap_at_10": overlap,
                "documents_scored": scored,
                "latency_seconds": latency,
            }
        )
    result = {
        "queries": len(queries),
        "exhaustive": exhaustive.__dict__,
        "champion_lists": {
            "champion_size": args.champion_size,
            "mean_overlap_at_10": sum(item["overlap_at_10"] for item in overlaps) / len(overlaps),
            "mean_documents_scored": sum(item["documents_scored"] for item in overlaps) / len(overlaps),
            "mean_latency_seconds": sum(item["latency_seconds"] for item in overlaps) / len(overlaps),
            "per_query": overlaps,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["exhaustive"], indent=2))


if __name__ == "__main__":
    main()
