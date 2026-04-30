# Data

This directory is intentionally empty in the repository: we do not
redistribute third-party datasets.

## Expected layout

```
data/
├── ord_qa/
│   ├── train.jsonl
│   └── dev.jsonl
├── hotpotqa/
│   ├── train.jsonl
│   └── dev.jsonl
├── paper_results.json       # machine-readable tables from the TCAD manuscript
├── exp1_version_shift/      # tool-version shift protocol notes
├── exp2_e2e_rag/            # end-to-end EDA RAG protocol notes
└── sample/
    └── toy.jsonl            # toy examples shipped with the repo for unit tests
```

## File format (listwise JSONL)

Each line is one query with one positive and `N-1` BM25 hard negatives:

```json
{
  "query": "How do I set a clock period in OpenROAD?",
  "positive": "create_clock -name clk -period 2.0 [get_ports clk]",
  "negatives": [
    "OpenROAD reads SDC files via read_sdc <file>.",
    "DEF stands for Design Exchange Format.",
    "..."
  ]
}
```

This format matches `bge-reranker` fine-tuning (see
[FlagEmbedding/examples](https://github.com/FlagOpen/FlagEmbedding))
and HotpotQA paragraph-pair preprocessing.

## Where to obtain the datasets

- **ORD-QA**: <https://github.com/lesliepy99/RAG-EDA> (Pu et al.,
  "Customized Retrieval Augmented Generation and Benchmarking for
  EDA Tool Documentation QA", ICCAD 2024).
- **HotpotQA**: <https://hotpotqa.github.io/>.

## Conversion scripts

This repository expects ORD-QA / HotpotQA data that has already been converted
to the listwise format above. Dataset-specific preprocessing is not bundled
because the raw datasets follow their original release formats and licenses.

## Paper result tables

`paper_results.json` mirrors the current TCAD manuscript tables, including the newly added confidence-control ablations, selection diagnostics, tool-version shift results, and end-to-end EDA RAG answer-quality results. Use:

```bash
python -m scripts.print_paper_results --table confidence_controls
```

## EDA-centric stress-test protocols

Protocol notes for the TCAD paper stress tests are under `data/exp1_version_shift/` and `data/exp2_e2e_rag/`. See `data/EDA_STRESS_TESTS.md`. The public companion repository intentionally does not include a one-shot runner, raw third-party sources, generated artifacts, generator outputs, or judge prompts.
