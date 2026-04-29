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
└── sample/
    └── toy.jsonl    # 16 examples, shipped with the repo for unit tests
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

Conversion scripts (`prep_ord_qa.py`, `prep_hotpotqa.py`) and
hard-negative mining utilities are part of the planned release
(see Release Roadmap in the top-level `README.md`).
