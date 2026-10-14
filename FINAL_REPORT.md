# Lexis Information Retrieval Report

## Collection
- Documents: 10788
- Tokens: 1468154
- Vocabulary: 30952
- Zipf fit: k=10893959.2931, exponent=-1.5820
- Heaps fit: k=26.6740, b=0.4991

## Ranking
| Split | Precision@10 | MAP | NDCG@10 |
|---|---:|---:|---:|
| train | 0.7475 | 0.1397 | 0.6204 |
| test | 0.6375 | 0.1865 | 0.5235 |

## Ablations
| Variant | MAP | MAP difference | p-value |
|---|---:|---:|---:|
| raw_tf | 0.0942 | -0.0455 | 0.0005 |
| without_idf | 0.1128 | -0.0269 | 0.0396 |
| without_length_normalization | 0.1602 | 0.0205 | 0.1253 |
| without_zone_weighting | 0.1445 | 0.0048 | 0.0949 |
| stemming | 0.1418 | 0.0021 | 0.3190 |
| stop_words | 0.1449 | 0.0052 | 0.0257 |

## Compression
Detailed machine-readable compression results are in `results/final_results.json`.

## Skip pointers
Eight term pairs were evaluated at three spacings each; detailed results are in `results/final_results.json`.

## Reproducibility
Generated with `python3 scripts/final_report.py`; training queries were used for ablations and test queries were evaluated separately.
