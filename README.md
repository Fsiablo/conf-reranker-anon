<h1 align="center">Conf-Reranker</h1>

<p align="center">
  <b>Confidence-Propagating Reranking for Trustworthy RAG in EDA</b><br>
  <i>Ranker--auditor training with risk-budgeted Top-k* context admission</i>
</p>

<p align="center">
  📄 <a href="#citation">Paper</a> &nbsp;|&nbsp;
  🧠 <a href="docs/method.md">Method Notes</a> &nbsp;|&nbsp;
  🚀 <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  📊 <a href="#results">Results</a> &nbsp;|&nbsp;
  🗺️ <a href="#release-roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-red.svg">
  <img src="https://img.shields.io/badge/license-MIT-green.svg">
  <img src="https://img.shields.io/badge/status-TCAD%20companion-blue.svg">
</p>

---

> **Status — Partial TCAD companion preview.** This repository accompanies the
> journal manuscript but is intentionally **not** a complete runnable
> reproduction package. It includes the core method skeleton, paper-facing
> result tables, and stress-test protocol notes. Pretrained checkpoints, raw
> third-party datasets, one-shot build scripts, generated artifacts, and judge
> prompts are not redistributed.

---

## 💡 What is Conf-Reranker?

Modern retrieval-augmented generation (RAG) pipelines commit to a fixed top-$k$
of retrieved passages, even when the reranker is **systematically overconfident**.
This is particularly painful in high-stakes domains such as **EDA tool documentation
QA**, where a single outdated PDK parameter can cause timing-closure failure
downstream.

**Conf-Reranker** addresses this with three coupled ideas:

1. **Dual-head cross-encoder.** A score head $s_i$ and a confidence head $c_i$
   share the encoder, with a stop-gradient on the confidence path so the ranker
   stays untouched.
2. **Ranker–auditor co-training.** A three-term objective —
   $\mathcal{L}_{\text{main}} + \lambda_c \mathcal{L}_{\text{conf}} + \lambda_r \mathcal{L}_{\text{reg}}$ —
   teaches the auditor to predict its own ranker's correctness without collapsing
   to a constant.
3. **Risk-Budgeted Top-$k^*$ inference.** Instead of fixed top-5, candidates are
   selected by utility $u_i = p_i \cdot c_i^\beta$ with an average-confidence
   gate $\rho$, adapting $k^*$ per query.

<p align="center">
  <img src="docs/assets/framework.png" width="85%"><br>
  <i>Figure 1 — Conf-Reranker framework: dual-head architecture with stop-gradient,
  co-trained with three-term loss, and Risk-Budgeted Top-k* inference.</i>
</p>

<p align="center">
  <img src="docs/assets/architecture.png" width="80%"><br>
  <i>Figure 2 — Architecture detail. The score and confidence heads share the
  encoder representation, adding about 0.05% parameter overhead on BGE-v2-m3.</i>
</p>

---

## 📊 Results

### Main results — ORD-QA & HotpotQA

Reranking performance (%, mean over 3 seeds). **Bold** = best within each backbone group.
$\Delta$ = absolute improvement over Fine-tuned; all $\Delta$nDCG@5 improvements are
statistically significant at $p<0.01$ by paired permutation test.

| Backbone | Variant | ORD-QA MRR@5 | ORD-QA nDCG@5 | ORD-QA R@5 | HotpotQA MRR@5 | HotpotQA nDCG@5 | HotpotQA R@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| DeBERTa-v3-base | Fine-tuned | 58.37 | 62.41 | 69.68 | 55.59 | 59.17 | 66.23 |
| DeBERTa-v3-base | **+ Conf-Reranker** | **59.84** | **64.15** | **71.95** | **57.26** | **61.34** | **68.71** |
| ELECTRA-base | Fine-tuned | 55.23 | 60.11 | 66.37 | 52.41 | 57.53 | 63.49 |
| ELECTRA-base | **+ Conf-Reranker** | **57.18** | **62.58** | **69.22** | **54.35** | **59.76** | **65.86** |
| BGE-reranker-large | Fine-tuned | 73.52 | 79.08 | 83.34 | 79.64 | 81.27 | 83.68 |
| BGE-reranker-large | **+ Conf-Reranker** | **74.89** | **80.73** | **85.12** | **80.92** | **82.51** | **84.53** |
| BGE-reranker-v2-m3 | Fine-tuned | 72.08 | 78.41 | 81.13 | 78.95 | 80.64 | 82.94 |
| BGE-reranker-v2-m3 | **+ Conf-Reranker** | **73.45** | **79.86** | **82.78** | **80.17** | **81.89** | **83.79** |

> **PLM gains:** $\Delta$nDCG@5 = **+1.74 ~ +2.47%**.
> **Dedicated reranker gains:** $\Delta$nDCG@5 = **+1.24 ~ +1.65%** (less headroom).
> Recall@1/3 numbers and per-seed values are reported in the supplementary material.

### Comparison with uncertainty / reranking baselines

On BGE-reranker-v2-m3, single A6000 (FP16). Latency in ms/query.

| Method | ORD-QA nDCG@5 | ORD-QA ECE | HotpotQA nDCG@5 | HotpotQA ECE | Params (M) | Latency (ms) | Mem (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned (top-5) | 78.41 | 0.12 | 80.64 | 0.11 | 568.0 | 7.8 | 6.2 |
| + Label Smoothing | 78.62 | 0.08 | 80.82 | 0.08 | 568.0 | 7.8 | 6.2 |
| + MC Dropout (T=10) | 78.74 | 0.07 | 80.93 | 0.07 | 568.0 | 41.2 | 6.2 |
| + Deep Ensembles (M=5) | 79.10 | 0.08 | 81.20 | 0.07 | 2840 | 25.3 | 31.0 |
| + Temperature Scaling | 78.90 | **0.05** | 81.00 | 0.06 | 568.0 | 7.9 | 6.2 |
| + Evidential Ranking | 78.97 | 0.07 | 81.12 | 0.07 | 568.1 | 8.0 | 6.2 |
| Self-RAG (rerank) | 78.83 | 0.10 | 80.87 | 0.10 | 568.0 | 46.5 | 6.2 |
| RankZephyr-7B distilled | 79.42 | 0.11 | 81.65 | 0.10 | 7000 | 152.0 | 14.0 |
| **Ours (fixed top-5)** | 79.63 | 0.06 | 81.72 | **0.05** | 568.3 | 8.0 | 6.3 |
| **Ours (Top-$k^*$, ρ=0.2)** | **79.86** | 0.06 | **81.89** | **0.05** | 568.3 | **8.1** | **6.3** |

> Conf-Reranker matches Temperature Scaling on ECE while gaining **+1.45 / +1.25 nDCG@5**
> on ORD-QA / HotpotQA, at essentially zero extra latency / memory.

### Score-derived confidence and stop-gradient controls

On BGE-reranker-v2-m3 / ORD-QA validation, the TCAD version adds controls for
score-derived confidence, calibrated-score selection, stop-gradient removal, and
inference-only confidence admission.

| Variant | Confidence source | Stop-grad. | Weighted train. | Top-k* | nDCG@5 | Δ | ECE | Stale@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned baseline | - | - | No | No | 78.41 | - | 0.120 | 6.8% |
| Sigmoid-score control | sigmoid(score) | N/A | No | Yes | 78.46 | +0.05 | 0.118 | 6.6% |
| Margin control | p1-p2 | N/A | No | Yes | 78.52 | +0.11 | 0.114 | 6.4% |
| Calibrated-score selective | Platt / isotonic | N/A | No | Yes | 78.58 | +0.17 | 0.083 | 6.3% |
| Conf head, no stop-gradient | Learned head | No | Yes | Yes | 77.92 | -0.49 | 0.094 | 5.9% |
| Conf head, inference-only | Learned head | Yes | No | Yes | 78.74 | +0.33 | 0.071 | 5.7% |
| Conf training, fixed top-5 | Learned head | Yes | Yes | No | 79.63 | +1.22 | 0.061 | 5.4% |
| **Full Conf-Reranker** | Learned head | Yes | Yes | Yes | **79.86** | **+1.45** | **0.058** | **4.7%** |

These controls show that the gain is not just a monotone score calibration:
score-derived confidence improves ECE but leaves ranking and stale admission
largely unchanged, while the learned stop-gradient auditor provides the main
benefit.

### OOD generalization (zero-shot cross-dataset)

<p align="center">
  <img src="docs/assets/ood_bar.png" width="55%"><br>
  <i>Figure 3 — Zero-shot cross-dataset OOD generalization on BGE-reranker-v2-m3.
  Conf-Reranker improves nDCG@5 by <b>+3.35</b> (ORD-QA→HotpotQA) and
  <b>+3.45</b> (HotpotQA→ORD-QA), 2–3× larger than in-distribution gains.</i>
</p>

### Calibration

<p align="center">
  <img src="docs/assets/calibration_curve.png" width="80%"><br>
  <i>Figure 4 — Reliability diagrams (ORD-QA left, HotpotQA right). Conf-Reranker
  (red) tracks the perfect-calibration diagonal; the fine-tuned baseline (blue)
  is systematically overconfident at high probabilities.</i>
</p>

### EDA-centric stress tests

The journal version includes two EDA-specific stress tests built from public
OpenROAD / ORD-QA artifacts. This repository documents the construction
protocols under `data/`, but does not ship the full runnable benchmark pipeline.

**Tool-version shift and stale-document admission**

| Method | nDCG@5 | Stale@5 | False admission | Mean k |
|---|---:|---:|---:|---:|
| Fine-tuned reranker | 74.62 | 58.4% | 30.7% | 5.00 |
| Temperature Scaling | 74.71 | 56.9% | 29.4% | 5.00 |
| Deep Ensembles (M=5) | 75.83 | 50.2% | 24.6% | 5.00 |
| **Conf-Reranker** | **79.18** | 28.7% | 14.3% | 3.86 |
| **Conf-Reranker + conformal filter** | 79.05 | **21.4%** | **8.7%** | 3.52 |

**End-to-end EDA RAG answer quality**

| Context selector | Correct | Unsupported | Wrong command | Valid script |
|---|---:|---:|---:|---:|
| BM25 top-5 | 44.3% | 31.6% | 25.4% | 58.7% |
| Fine-tuned reranker top-5 | 54.1% | 23.8% | 17.6% | 70.9% |
| Temperature-scaled top-5 | 55.2% | 22.7% | 16.9% | 71.8% |
| **Conf-Reranker Top-k*** | **61.7%** | 13.5% | 8.4% | 82.1% |
| **Conf-Reranker + conformal filter** | 61.3% | **10.8%** | **6.1%** | **84.6%** |

### Case study: Noisy EDA candidate reranking

<p align="center">
  <img src="docs/assets/eda-case1-scores.png" width="65%"><br>
  <i>Figure 5 — On a 7nm FinFET routing-delay query, two high-relevance but
  low-confidence candidates (an outdated PDK timing model, unverifiable forum
  claims) are filtered by the ρ=0.2 confidence gate, preventing contaminated
  evidence from leaking into downstream generation.</i>
</p>

For complete tables, see [`docs/results.md`](docs/results.md) or the
machine-readable file [`data/paper_results.json`](data/paper_results.json).

---

## 📦 Repository Layout

```
conf-reranker/
├── conf_reranker/          # Core library (importable)
│   ├── model.py            # Dual-head cross-encoder (Eq. 1-2)
│   ├── loss.py             # L_main + λ_c L_conf + λ_r L_reg (Eq. 3-6)
│   ├── inference.py        # Risk-Budgeted Top-k* (Algorithm 1)
│   ├── trainer.py          # Co-training loop skeleton
│   └── data.py             # JSONL dataset + collator
├── scripts/
│   ├── demo.py                 # 60s CPU demo (try this first!)
│   ├── run_train.py            # Training entry
│   ├── run_eval.py             # Evaluation entry
│   ├── print_paper_results.py  # Print data/paper_results.json as Markdown
│   ├── train.sh                # Example launch script
│   └── eval.sh
├── configs/
│   ├── default.yaml            # Defaults (λ_c=0.9, λ_r=0.2, ρ=0.2, β=2)
│   ├── ord_qa.yaml             # ORD-QA preset
│   ├── hotpotqa.yaml           # HotpotQA preset
│   └── ablation_controls.yaml  # Score-derived confidence / stop-gradient controls
├── data/
│   ├── README.md               # Data prep instructions
│   ├── EDA_STRESS_TESTS.md     # Tool-version and e2e RAG stress-test protocols
│   ├── paper_results.json      # Machine-readable tables from the manuscript
│   ├── exp1_version_shift/     # Version-shift protocol notes
│   ├── exp2_e2e_rag/           # E2E query-pack protocol notes
│   └── sample/toy.jsonl        # Toy dataset for tests/demo
├── tests/                      # Smoke tests
├── docs/
│   ├── method.md               # Method walk-through
│   ├── results.md              # Paper tables mirrored in Markdown
│   └── assets/                 # Figures (shared with paper)
├── requirements.txt
├── setup.py
├── LICENSE                 # MIT
└── CITATION.cff
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/anonymous/conf-reranker.git    # anonymized
cd conf-reranker
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e .                                    # installs core deps
```

### Run the 60-second demo

The demo loads a small public cross-encoder, builds the dual-head wrapper, and
runs Risk-Budgeted Top-$k^*$ inference on a toy 4-passage example. **No GPU,
no dataset download required.**

```bash
python -m scripts.demo
```

Expected output (truncated):

```
Query: "Predict routing delay of critical path in 7nm FinFET..."
Candidates after dual-head scoring:
  d1   s=0.91  c=0.79   utility=0.431
  d2   s=0.84  c=0.42   utility=0.119
  d3   s=0.82  c=0.31   utility=0.067
  d4   s=0.68  c=0.84   utility=0.308
Risk-Budgeted Top-k* (ρ=0.2):
  Selected: [d1, d4]   (k*=2, mean confidence=0.815 >= 1-ρ)
```

### Programmatic API

```python
import torch
from conf_reranker import ConfReranker, risk_budgeted_topk
from conf_reranker.model import ConfRerankerConfig
from conf_reranker.inference import RiskBudgetConfig
from transformers import AutoTokenizer

cfg = ConfRerankerConfig(backbone_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
model = ConfReranker(cfg).eval()
tok = AutoTokenizer.from_pretrained(cfg.backbone_name)

scores, confs = model.score("query text", ["doc 1", "doc 2", "doc 3"], tokenizer=tok)
selected, utility, _, low_conf = risk_budgeted_topk(
    scores, confs, RiskBudgetConfig(rho=0.2, beta=2.0)
)
```

### Reproducing the paper

This public package is **not** intended to be a complete runnable reproduction
of the paper. It provides three lightweight audit layers instead:

1. **Core method skeleton.** `conf_reranker/`, `configs/default.yaml`, and the
   training/evaluation entry points document the model, loss, Top-k* admission,
   and conformal threshold utility.
2. **Paper result tables.** `data/paper_results.json` stores the headline
   result tables, including the confidence-control ablations added in the TCAD
   version. Print them with:

   ```bash
   python -m scripts.print_paper_results --table confidence_controls
   python -m scripts.print_paper_results --table version_shift
   ```

3. **EDA stress-test protocols.** `data/EDA_STRESS_TESTS.md` documents the
   construction logic and expected artifact names, but the one-shot runner,
   raw sources, generated artifacts, generator outputs, and judge prompts are
   intentionally not included.

Pretrained checkpoints and raw third-party datasets are not redistributed; see
`data/README.md` for the expected JSONL layout.

---

## 🗺️ Release Roadmap

| Milestone | Status |
|---|---|
| Core method skeleton (model / loss / inference) | ✅ Released |
| Demo & toy data | ✅ Released |
| Documentation (method and result tables) | ✅ Released |
| Machine-readable paper result tables | ✅ Released |
| EDA stress-test protocol notes | ✅ Released |
| Confidence-control ablation configuration | ✅ Released |
| Pretrained checkpoints (BGE-v2-m3, DeBERTa-v3) | Not included |
| Full ORD-QA / HotpotQA preprocessing pipeline | Not included |
| One-shot EDA stress-test runner and generated artifacts | Not included |
| Multi-backbone raw training logs | Not included |

---

## 📚 Datasets

| Dataset | Source | Used For |
|---|---|---|
| ORD-QA | [Pu et al., 2024](https://github.com/lesliepy99/RAG-EDA) | EDA documentation QA (in-domain) |
| HotpotQA (fullwiki) | [Yang et al., 2018](https://hotpotqa.github.io/) | Open-domain multi-hop QA (OOD) |
| OpenROAD documentation snapshots | [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) | Tool-version shift and command-whitelist construction |

We do **not** redistribute the datasets. Please follow each dataset's original
license and download instructions; see [`data/README.md`](data/README.md) for the
expected JSONL layout.

---

## 🤝 Acknowledgements

This work builds on
[BGE](https://github.com/FlagOpen/FlagEmbedding) reranker family,
[ORD-QA / RAG-EDA](https://github.com/lesliepy99/RAG-EDA),
and the [HotpotQA](https://hotpotqa.github.io/) benchmark.
We thank the authors of these resources for releasing them publicly.

---

## 📖 Citation

If you use this code or method in your research, please cite our paper:

```bibtex
@article{confreranker,
  title   = {Confidence-Propagating Reranking for Trustworthy
             Retrieval-Augmented Generation in Electronic Design Automation},
  author  = {Anonymous Authors},
  journal = {IEEE Transactions on Computer-Aided Design of Integrated
             Circuits and Systems (TCAD)},
  year    = {to appear},
  note    = {Preprint.}
}
```

> ℹ️ The BibTeX above will be updated with the final author list, volume, and
> page numbers once the paper is published.

---

## 📝 License

This project is released under the [MIT License](LICENSE). The pretrained
checkpoints (when released) will be subject to the underlying backbones'
respective licenses (Apache-2.0 for BGE family, MIT for DeBERTa).

---

## 📮 Contact

For questions about the method, please open a GitHub Issue. For questions
specific to the editorial process, please contact the corresponding author
through the journal editorial system.
