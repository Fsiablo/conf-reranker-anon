# Method Notes — Conf-Reranker

This document expands on the method behind Conf-Reranker, complementing the
formal exposition in **Section IV** of the paper. It is meant for readers who
want a one-page mental model before reading the code in `conf_reranker/`.

---

## 1. Problem

Given a query $q$ and a candidate set $\mathcal{C} = \{d_1, \dots, d_N\}$ from a
first-stage retriever, a reranker produces an ordering. The dominant paradigm
fine-tunes a cross-encoder with cross-entropy on the relevance label and then
returns a fixed top-$k$ (typically $k=5$).

Two failure modes consistently appear in high-stakes RAG (EDA documentation,
medical QA, code search):

1. **Systematic overconfidence.** The reranker assigns near-1.0 probabilities to
   candidates that are partially relevant, outdated, or unverifiable.
2. **Inflexible cutoff.** Returning a fixed top-5 forwards low-confidence
   passages even when the reliable evidence is fewer than 5 items, contaminating
   downstream generation.

---

## 2. Dual-Head Cross-Encoder

We add a small **confidence head** $h_c$ on top of the encoder pooled
representation, alongside the standard score head $h_s$:

$$
s_i = h_s(\text{enc}(q, d_i)), \quad c_i = \sigma(h_c(\text{sg}[\text{enc}(q, d_i)]))
$$

The `sg[·]` is **stop-gradient**: gradients from the confidence loss do not flow
back into the encoder when training the confidence head. This keeps the ranker
behavior *invariant* to the addition of the auditor — a property we found
critical to avoid the confidence head dragging down ranking accuracy.

The parameter overhead is **<0.05%** of the backbone (a 0.3M-parameter linear-
GELU-linear MLP for BGE-large).

See `conf_reranker/model.py`.

---

## 3. Three-Term Loss

The training objective combines three terms:

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{main}}
            \;+\; \lambda_c \mathcal{L}_{\text{conf}}
            \;+\; \lambda_r \mathcal{L}_{\text{reg}}
$$

| Term | Role | Form |
|---|---|---|
| $\mathcal{L}_{\text{main}}$ | Standard listwise ranking loss on $s_i$ | Cross-entropy over softmax($s/T_0$) |
| $\mathcal{L}_{\text{conf}}$ | Auditor learns to predict its own ranker's correctness | BCE between $c_i$ and a soft target derived from per-candidate Spearman agreement |
| $\mathcal{L}_{\text{reg}}$ | Anti-collapse regularizer (prevents $c_i \to$ const) | Negative entropy of the batch-mean confidence + $\ell_2$ |

Defaults: $\lambda_c = 0.9$, $\lambda_r = 0.2$, $T_0 = 0.8$, $\beta = 2$.

See `conf_reranker/loss.py`.

---

## 4. Risk-Budgeted Top-$k^*$ Inference

At inference, we score each candidate's **utility**:

$$
u_i \;=\; p_i \cdot c_i^{\beta},
\quad p_i = \mathrm{softmax}(s)_i
$$

and select the smallest prefix $\mathcal{S}_{k^*}$ (in utility-sorted order) such
that the **average confidence within the prefix** meets the budget $\rho$:

$$
k^{*} \;=\; \min\Bigl\{k \;:\;
  \tfrac{1}{k}\textstyle\sum_{i \in \mathcal{S}_k} c_i \;\ge\; 1 - \rho \Bigr\}
$$

This adapts $k$ per query: easy queries with many high-confidence candidates
get a large $k^*$; ambiguous queries get a small $k^*$. Empirically the mean
$k^*$ is **4.21 / 4.47** on ORD-QA / HotpotQA at $\rho = 0.2$ — close to but
not equal to the conventional fixed top-5.

See `conf_reranker/inference.py`.

---

## 5. Why this is not "just calibration"

Post-hoc calibration (Temperature Scaling) reduces ECE but **cannot change the
ranking order**. Conf-Reranker improves both:

- nDCG@5 gain: **+1.45 / +1.25** on ORD-QA / HotpotQA (vs. TS: +0.49 / +0.36)
- ECE: matches TS (0.06 / 0.05 vs. 0.05 / 0.06)
- Latency: **+0.3 ms** vs. baseline (vs. Deep Ensembles: +17.5 ms, 5× memory)

The OOD effect is even larger: $+3.35$ / $+3.45$ nDCG@5 in the zero-shot
cross-dataset setting, suggesting reliability blindness is *primarily a
distributional-robustness problem* rather than a calibration problem.

---

## 6. Pointers

| To understand... | Read |
|---|---|
| The model architecture | `conf_reranker/model.py` + Section IV-A of paper |
| The loss derivation | `conf_reranker/loss.py` + Section IV-B of paper |
| The inference algorithm | `conf_reranker/inference.py` + Algorithm 1 of paper |
| The theoretical guarantees | Section V of paper (variance reduction + Platt fixed point) |
| The full ablation table | Section VI-C of paper + supplementary material |
