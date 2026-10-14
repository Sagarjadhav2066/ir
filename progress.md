# Lexis Information Retrieval Assignment Progress

## Problem statement

Build **Lexis**, a search engine over the supplied Reuters news archive. The
system must support:

- Boolean queries with `AND`, `OR`, `NOT`, parentheses, quoted phrases, and
  proximity queries such as `economy /5 growth`.
- Wildcard queries containing one or more `*` characters.
- Spelling correction suggestions for query terms absent from the vocabulary.
- Free-text ranked retrieval using cosine similarity over TF-IDF vectors.
- Separate title and body weights, returning the top 10 ranked documents.

The implementation must also:

- Build a positional inverted index at scale using streamed documents,
  disk-backed partial indexes, merging, and four parallel shards.
- Support additions and deletions without a complete rebuild.
- Compress the dictionary and gap-encoded postings.
- Apply one documented tokenizer and normalizer consistently to documents and
  queries.
- Match a lowercased brute-force reference implementation for exact retrieval
  when stemming is disabled.

The final report must include:

- Zipf and Heaps fits, including comparison with textbook Reuters-RCV1 values.
- Skip-pointer savings for at least eight term pairs and three spacings.
- Positional-index size compared with a plain index.
- SPIMI versus BSBI and indexing results for at least four memory limits.
- Four-shard speedup.
- Compression results for frequent and rare terms.
- Precision@10, an 11-point precision-recall curve, MAP, and NDCG@10.
- Comparisons between the Boolean system, raw-TF baseline, and best system.
- At least two pruning methods with overlap@10, documents scored, and latency.
- Ablation MAP changes and p-values for stemming, stop words, IDF, length
  normalization, and zone weighting.

## Supplied data

| Path | Purpose |
|---|---|
| `corpus/` | 10,788 Reuters articles; title, blank line, and body |
| `queries_train.tsv` | 40 queries for tuning |
| `queries_test.tsv` | 40 queries for final evaluation |
| `qrels.tsv` | Graded relevance judgments |
| `boolean_tests.tsv` | Expected Boolean, phrase, wildcard, and proximity results |
| `misspellings.tsv` | 60 misspelled terms and intended corrections |

## Implementation plan

1. Validate the corpus and all TSV files.
2. Implement shared tokenization and normalization.
3. Build a brute-force reference evaluator for correctness.
4. Implement the positional inverted index and basic retrieval.
5. Add Boolean parsing, phrase matching, and proximity matching.
6. Add SPIMI and BSBI disk-backed construction.
7. Add four-shard parallel construction and incremental segments/tombstones.
8. Add compressed postings, dictionary compression, and skip pointers.
9. Add wildcard expansion and spelling correction.
10. Add title/body TF-IDF ranking and top-10 retrieval.
11. Add pruning techniques and efficiency measurements.
12. Tune on training queries and evaluate once on test queries.
13. Run ablations and statistical significance tests.
14. Generate reproducible tables, plots, and the final report.

## Status

### Completed

- Read and extracted the assignment requirements.
- Confirmed the corpus contains 10,788 documents.
- Confirmed the corpus format using a sample article.
- Inspected the query, qrels, Boolean-test, and misspelling TSV formats.
- Created the project progress tracker.
- Implemented and passed corpus/TSV validation in `scripts/validate_dataset.py`.
- Implemented shared case-folding, tokenization, stop-word configuration, and
  corpus title/body parsing in `lexis/text.py`.
- Implemented the brute-force query parser and reference evaluator in
  `lexis/reference.py`.
- Passed all 25 supplied exact retrieval cases, including 10 Boolean, 7 phrase,
  4 proximity, and 4 wildcard queries.
- Implemented the in-memory positional inverted index in `lexis/index.py`,
  including document positions, title/body frequencies, wildcard expansion,
  phrase matching, proximity matching, Boolean evaluation, and active-document
  filtering.
- Verified the positional index against the brute-force evaluator: all 25
  supplied exact retrieval cases pass.
- Implemented disk-backed construction in `lexis/scalable.py` with streamed
  partial blocks, deterministic merging, memory-limit controls, and
  configurable shard workers.
- Added `scripts/build_index.py` for reproducible SPIMI/BSBI build commands and
  JSON metrics.
- Validated a four-shard build: 10,788 documents, 178 blocks, and all 25 exact
  retrieval cases passed.
- Measured single-shard 5,000-term builds: 39 blocks and about 138 MB written
  for both the SPIMI and BSBI benchmark runs; elapsed times were 10.75 and
  10.88 seconds respectively in this environment.
- Added a distinct BSBI pair-buffer/sort path; at the same 5,000-unit limit it
  produced 163 pair blocks versus 39 SPIMI term blocks.
- Added gap, variable-byte, and gamma encoding utilities in
  `lexis/compression.py`, with round-trip validation for variable-byte coding.
- Added skip-pointer intersection measurement utilities in `lexis/skip.py`.
- Added incremental `add_document` and logical/physical `delete_document`
  operations to the positional index and validated both behaviors.
- Implemented title/body-weighted TF-IDF cosine ranking in `lexis/ranking.py`.
- Added raw-TF, IDF, length-normalization, and zone-weighting configurations.
- Added precision@10, MAP, and NDCG@10 evaluation against graded qrels.
- Added champion-list pruning measurements with overlap@10, documents scored,
  and latency in `scripts/evaluate_ranking.py`.
- Training ranking baseline currently measures precision@10 `0.7475`, MAP
  `0.13968`, and NDCG@10 `0.62043`; the test split measures precision@10
  `0.6375`, MAP `0.18653`, and NDCG@10 `0.52352`.
- Training ablations were measured for raw TF, no IDF, no length normalization,
  and no zone weighting.
- Added a deterministic stemming option and stop-word option to the shared
  tokenizer.
- Generated `FINAL_REPORT.md` and `results/final_results.json` with:
  collection statistics, Zipf/Heaps fits, train/test ranking metrics,
  six MAP ablations with paired p-values, compression measurements, and
  24 skip-pointer measurements (8 term pairs at 3 spacings).

### Completed

- Statistical ablations, stemming/stop-word variants, and final report
  generation.

### Not started

- None.

## Validation policy

Every optimized retrieval component will be checked against the brute-force
reference implementation before performance measurements are accepted. Training
queries will be used for tuning; test queries will be reserved for the final
evaluation.

## Final generated artifacts

- `FINAL_REPORT.md`
- `results/final_results.json`
- `scripts/validate_dataset.py`
- `scripts/build_index.py`
- `scripts/evaluate_ranking.py`
