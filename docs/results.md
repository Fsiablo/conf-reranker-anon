# Paper Results

This page mirrors the TCAD journal manuscript tables. The same values are available in machine-readable form at `data/paper_results.json`.

## Main reranking results

Reranking performance is reported in percent and averaged over three training seeds.

| Backbone | Variant | ORD-QA MRR@5 | ORD-QA nDCG@5 | ORD-QA R@5 | HotpotQA MRR@5 | HotpotQA nDCG@5 | HotpotQA R@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| DeBERTa-v3-base | Fine-tuned | 58.37 | 62.41 | 69.68 | 55.59 | 59.17 | 66.23 |
| DeBERTa-v3-base | Conf-Reranker | **59.84** | **64.15** | **71.95** | **57.26** | **61.34** | **68.71** |
| ELECTRA-base | Fine-tuned | 55.23 | 60.11 | 66.37 | 52.41 | 57.53 | 63.49 |
| ELECTRA-base | Conf-Reranker | **57.18** | **62.58** | **69.22** | **54.35** | **59.76** | **65.86** |
| BGE-reranker-large | Fine-tuned | 73.52 | 79.08 | 83.34 | 79.64 | 81.27 | 83.68 |
| BGE-reranker-large | Conf-Reranker | **74.89** | **80.73** | **85.12** | **80.92** | **82.51** | **84.53** |
| BGE-reranker-v2-m3 | Fine-tuned | 72.08 | 78.41 | 81.13 | 78.95 | 80.64 | 82.94 |
| BGE-reranker-v2-m3 | Conf-Reranker | **73.45** | **79.86** | **82.78** | **80.17** | **81.89** | **83.79** |

## Confidence-control ablations

These controls isolate score calibration, score-margin uncertainty, stop-gradient decoupling, weighted training, and adaptive Top-k* admission.

| Variant | Confidence source | Stop-grad. | Weighted train. | Top-k* | nDCG@5 | Delta | ECE | Stale@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned baseline | - | - | No | No | 78.41 | - | 0.120 | 6.8% |
| Sigmoid-score control | sigmoid(score) | N/A | No | Yes | 78.46 | +0.05 | 0.118 | 6.6% |
| Margin control | p1-p2 | N/A | No | Yes | 78.52 | +0.11 | 0.114 | 6.4% |
| Calibrated-score selective | Platt / isotonic | N/A | No | Yes | 78.58 | +0.17 | 0.083 | 6.3% |
| Conf head, no stop-gradient | Learned head | No | Yes | Yes | 77.92 | -0.49 | 0.094 | 5.9% |
| Conf head, inference-only | Learned head | Yes | No | Yes | 78.74 | +0.33 | 0.071 | 5.7% |
| Conf training, fixed top-5 | Learned head | Yes | Yes | No | 79.63 | +1.22 | 0.061 | 5.4% |
| Full Conf-Reranker | Learned head | Yes | Yes | Yes | **79.86** | **+1.45** | **0.058** | **4.7%** |

The `Stale@5` values in this table are in-distribution ORD-QA diagnostics computed from version labels in the candidate pool. They are not directly comparable to the much larger `Stale@5` values in the explicit stale-injection stress test below.

## Selection diagnostics

| Dataset | nDCG@5 | ECE | Mean k* | Context reduction |
|---|---:|---:|---:|---:|
| ORD-QA | 79.86 | 0.06 | 4.21 | 15.8% |
| HotpotQA | 81.89 | 0.05 | 4.47 | 10.6% |

## EDA tool-version shift

| Method | nDCG@5 | Stale@5 | False admission | Mean k |
|---|---:|---:|---:|---:|
| Fine-tuned reranker | 74.62 | 58.4% | 30.7% | 5.00 |
| Temperature Scaling | 74.71 | 56.9% | 29.4% | 5.00 |
| Deep Ensembles (M=5) | 75.83 | 50.2% | 24.6% | 5.00 |
| Conf-Reranker | **79.18** | 28.7% | 14.3% | 3.86 |
| Conf-Reranker + conformal filter | 79.05 | **21.4%** | **8.7%** | 3.52 |

## End-to-end EDA RAG answer quality

| Context selector | Correct | Unsupported | Wrong command | Valid script |
|---|---:|---:|---:|---:|
| BM25 top-5 | 44.3% | 31.6% | 25.4% | 58.7% |
| Fine-tuned reranker top-5 | 54.1% | 23.8% | 17.6% | 70.9% |
| Temperature-scaled top-5 | 55.2% | 22.7% | 16.9% | 71.8% |
| Conf-Reranker Top-k* | **61.7%** | 13.5% | 8.4% | 82.1% |
| Conf-Reranker + conformal filter | 61.3% | **10.8%** | **6.1%** | **84.6%** |
