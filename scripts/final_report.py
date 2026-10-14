#!/usr/bin/env python3
"""Generate the final reproducible Lexis assignment report."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.compression import compression_sizes
from lexis.index import PositionalIndex
from lexis.ranking import RankingConfig, RankedSearcher, evaluate, read_qrels, read_queries
from lexis.skip import intersect_with_metrics, recommended_spacings
from lexis.text import TextProcessor, parse_document


ROOT = Path(__file__).resolve().parents[1]


def fit_log(values: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(values)
    sx = sum(x for x, _ in values)
    sy = sum(y for _, y in values)
    sxx = sum(x * x for x, _ in values)
    sxy = sum(x * y for x, y in values)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return math.exp(intercept), slope


def p_value(differences: list[float]) -> float:
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / max(1, len(differences) - 1)
    if variance == 0:
        return 0.0 if mean else 1.0
    z = abs(mean) / math.sqrt(variance / len(differences))
    return math.erfc(z / math.sqrt(2))


def query_aps(searcher: RankedSearcher, queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> list[float]:
    values = []
    for query_id, query in queries.items():
        ranked = [item.doc_id for item in searcher.search(query)]
        relevant = {doc for doc, grade in qrels.get(query_id, {}).items() if grade > 0}
        hits = total = 0.0
        for rank, doc in enumerate(ranked, 1):
            if doc in relevant:
                hits += 1
                total += hits / rank
        values.append(total / len(relevant) if relevant else 0.0)
    return values


def main() -> None:
    corpus = ROOT / "corpus"
    train = read_queries(ROOT / "queries_train.tsv")
    test = read_queries(ROOT / "queries_test.tsv")
    qrels = read_qrels(ROOT / "qrels.tsv")
    index = PositionalIndex.from_corpus(corpus)
    baseline = RankedSearcher(index)
    metrics = {"train": evaluate(baseline, train, qrels).__dict__, "test": evaluate(baseline, test, qrels).__dict__}

    ablations = {}
    for name, config in {
        "raw_tf": RankingConfig(raw_tf=True, use_idf=False),
        "without_idf": RankingConfig(use_idf=False),
        "without_length_normalization": RankingConfig(length_normalization=False),
        "without_zone_weighting": RankingConfig(title_weight=1.0),
    }.items():
        candidate = RankedSearcher(index, config)
        base_ap = query_aps(baseline, train, qrels)
        candidate_ap = query_aps(candidate, train, qrels)
        ablations[name] = {
            "metrics": evaluate(candidate, train, qrels).__dict__,
            "map_difference": sum(candidate_ap) / len(candidate_ap) - sum(base_ap) / len(base_ap),
            "p_value": p_value([a - b for a, b in zip(candidate_ap, base_ap)]),
        }

    for name, processor in {
        "stemming": TextProcessor(stemming=True),
        "stop_words": TextProcessor(remove_stop_words=True),
    }.items():
        variant = RankedSearcher(PositionalIndex.from_corpus(corpus, processor))
        base_ap = query_aps(baseline, train, qrels)
        variant_ap = query_aps(variant, train, qrels)
        ablations[name] = {
            "metrics": evaluate(variant, train, qrels).__dict__,
            "map_difference": sum(variant_ap) / len(variant_ap) - sum(base_ap) / len(base_ap),
            "p_value": p_value([a - b for a, b in zip(variant_ap, base_ap)]),
        }

    frequencies = Counter()
    total_tokens = 0
    cumulative = []
    vocabulary = set()
    for path in sorted(corpus.glob("*.txt")):
        document = parse_document(path.stem, path.read_text(encoding="utf-8"))
        tokens = document.all_tokens
        frequencies.update(tokens)
        vocabulary.update(tokens)
        total_tokens += len(tokens)
        cumulative.append((total_tokens, len(vocabulary)))
    zipf_values = [(math.log(rank), math.log(freq)) for rank, freq in enumerate(sorted(frequencies.values(), reverse=True), 1)]
    zipf_k, zipf_exponent = fit_log(zipf_values)
    heaps_values = [(math.log(tokens), math.log(types)) for tokens, types in cumulative if tokens and types]
    heaps_k, heaps_b = fit_log(heaps_values)

    frequent = [term for term, _ in frequencies.most_common(100)]
    rare = [term for term, _ in sorted(frequencies.items(), key=lambda item: item[1])[:10000]]
    compression = {}
    for label, terms in {"top_100": frequent, "rarest_10000": rare}.items():
        ids = sorted(
            {int(posting.doc_id.split("_")[1]) for term in terms for posting in index.postings[term]}
        )
        compression[label] = compression_sizes(ids) if ids else {}

    terms_by_df = sorted(index.vocabulary, key=lambda term: len(index.postings[term]), reverse=True)
    skip_results = []
    for left, right in zip(terms_by_df[:16:2], terms_by_df[1:16:2]):
        left_ids = [int(item.doc_id.split("_")[1]) for item in index.postings[left]]
        right_ids = [int(item.doc_id.split("_")[1]) for item in index.postings[right]]
        for spacing in recommended_spacings(max(len(left_ids), len(right_ids))):
            _, result = intersect_with_metrics(left_ids, right_ids, spacing)
            skip_results.append({"left": left, "right": right, **result.__dict__})

    report = {
        "collection": {"documents": len(index.document_ids), "tokens": total_tokens, "vocabulary": len(vocabulary)},
        "zipf": {"k": zipf_k, "exponent": zipf_exponent},
        "heaps": {"k": heaps_k, "b": heaps_b},
        "ranking": metrics,
        "ablations": ablations,
        "compression": compression,
        "skip_pointers": skip_results,
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "final_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Lexis Information Retrieval Report",
        "",
        "## Collection",
        f"- Documents: {len(index.document_ids)}",
        f"- Tokens: {total_tokens}",
        f"- Vocabulary: {len(vocabulary)}",
        f"- Zipf fit: k={zipf_k:.4f}, exponent={zipf_exponent:.4f}",
        f"- Heaps fit: k={heaps_k:.4f}, b={heaps_b:.4f}",
        "",
        "## Ranking",
        "| Split | Precision@10 | MAP | NDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for split, values in metrics.items():
        lines.append(f"| {split} | {values['precision_at_10']:.4f} | {values['map']:.4f} | {values['ndcg_at_10']:.4f} |")
    lines += ["", "## Ablations", "| Variant | MAP | MAP difference | p-value |", "|---|---:|---:|---:|"]
    for name, values in ablations.items():
        lines.append(f"| {name} | {values['metrics']['map']:.4f} | {values['map_difference']:.4f} | {values['p_value']:.4f} |")
    lines += ["", "## Compression", "Detailed machine-readable compression results are in `results/final_results.json`.", "", "## Skip pointers", "Eight term pairs were evaluated at three spacings each; detailed results are in `results/final_results.json`.", "", "## Reproducibility", "Generated with `python3 scripts/final_report.py`; training queries were used for ablations and test queries were evaluated separately."]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote FINAL_REPORT.md and results/final_results.json")


if __name__ == "__main__":
    main()
