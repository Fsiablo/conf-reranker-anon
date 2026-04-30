# Method Notes — Conf-Reranker

This document tracks the TCAD journal version of the paper and the code in `conf_reranker/`.

## 1. Problem

Given a query \(q\) and a first-stage candidate set \(\{d_i\}_{i=1}^N\), a cross-encoder reranker typically returns a scalar relevance score and a fixed top-\(k\) context. In EDA RAG this is not enough: a stale command page can be lexically relevant, enter the context window, and lead the generator toward an obsolete Tcl or SDC recommendation. The paper calls this reranking-stage failure mode **reliability-blindness**.

## 2. Dual-head cross-encoder

Conf-Reranker keeps the backbone encoder and adds two lightweight heads on the pooled representation:

\[
s_i = f_{\mathrm{score}}(h_i), \qquad
c_i = \sigma\!\left(f_{\mathrm{conf}}(\operatorname{sg}(h_i))\right).
\]

The confidence path receives a stop-gradient copy of the encoder representation. Therefore, gradients from the confidence head do not update the encoder. The confidence loss also uses a detached copy of \(\sigma(s_i)\), so the self-consistency term updates only the auditor. This asymmetric coupling lets the auditor track the ranker without pulling the ranker toward an easy confidence target.

The default head bottleneck is \(d/4\), matching the paper. For BGE-reranker-v2-m3 this adds about 0.3M parameters, roughly 0.05% of the full model.

Code: `conf_reranker/model.py`.

## 3. Ranker--auditor objective

Training uses the three-term objective

\[
\mathcal{L} = \mathcal{L}_{\mathrm{main}} + \lambda_c \mathcal{L}_{\mathrm{conf}} + \lambda_r \mathcal{L}_{\mathrm{reg}}.
\]

| Term | Role | Implemented form |
|---|---|---|
| \(\mathcal{L}_{\mathrm{main}}\) | Listwise ranking with auditor weights | \(-\sum_i \operatorname{sg}(c_i)y_i\log \mathrm{softmax}(s)_i\) |
| \(\mathcal{L}_{\mathrm{conf}}\) | Self-consistency target | \(\sum_i (c_i - \operatorname{sg}(\sigma(s_i)))^2\) |
| \(\mathcal{L}_{\mathrm{reg}}\) | Prevent premature saturation | \(\sum_i c_i\log c_i + (1-c_i)\log(1-c_i)\) |

Defaults are \(\lambda_c=0.9\), \(\lambda_r=0.2\), \(T_0=0.8\), \(\beta=2\), and \(\rho=0.2\).

Code: `conf_reranker/loss.py`.

## 4. Risk-budgeted Top-\(k^*\) inference

At inference, raw logits are confidence-temperature scaled:

\[
s'_i = s_i(c_i+\varepsilon)/T_0,
\qquad p_i = \mathrm{softmax}(s')_i,
\qquad u_i = p_i c_i^\beta.
\]

Candidates are sorted by utility. The selected prefix is the smallest \(k\) whose average confidence satisfies the risk budget:

\[
k^* = \min\left\{ k : \frac{1}{k}\sum_{i\in \mathcal{S}_k} c_i \ge 1-\rho \right\}.
\]

If no prefix satisfies the budget, the implementation returns an empty selection with `low_conf_flag=True`; downstream code should abstain, request human review, or use a no-context fallback.

Code: `conf_reranker/inference.py`.

## 5. Split-conformal filter

The optional conformal filter calibrates a threshold on irrelevant calibration candidates. Given held-out confidence scores and binary correctness labels, `conformal_threshold` returns the upper \(1-\alpha\) quantile of confidences assigned to irrelevant candidates. At deployment, compose Top-\(k^*\) with `c_i >= tau`. Under the exchangeability assumption conditional on irrelevance, this controls the marginal false-admission probability by \(\alpha + 1/(M_0+1)\).

## 6. Why this is not just calibration

The current paper includes targeted controls on BGE-reranker-v2-m3 / ORD-QA:

| Variant | nDCG@5 | ECE | Stale@5 |
|---|---:|---:|---:|
| Fine-tuned baseline | 78.41 | 0.120 | 6.8% |
| Sigmoid-score control | 78.46 | 0.118 | 6.6% |
| Margin control | 78.52 | 0.114 | 6.4% |
| Calibrated-score selective | 78.58 | 0.083 | 6.3% |
| Conf head, no stop-gradient | 77.92 | 0.094 | 5.9% |
| Conf head, inference-only | 78.74 | 0.071 | 5.7% |
| Conf training, fixed top-5 | 79.63 | 0.061 | 5.4% |
| Full Conf-Reranker | **79.86** | **0.058** | **4.7%** |

Score-derived controls improve calibration but barely move ranking or stale admission. The learned auditor and asymmetric training are therefore not just a monotone rescaling of the ranker score.

## 7. Pointers

| To understand... | Read |
|---|---|
| Model architecture | `conf_reranker/model.py` |
| Training loss | `conf_reranker/loss.py` |
| Top-\(k^*\) and conformal filtering | `conf_reranker/inference.py` |
| Paper result tables | `docs/results.md` and `data/paper_results.json` |
| EDA stress-test construction | `data/EDA_STRESS_TESTS.md` |
