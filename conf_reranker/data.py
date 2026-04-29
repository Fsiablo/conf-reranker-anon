"""Listwise dataset utilities.

Each training example is a listwise group ``(query, [doc_1, ..., doc_N])``
with binary relevance labels ``y in {0, 1}^N``. The first document is the
positive; the remaining N-1 are BM25 hard negatives.

The data loader expects a JSONL file with one example per line:

    {"query": "...", "positive": "...", "negatives": ["...", "...", ...]}

This is identical in spirit to the bge-reranker fine-tuning format and
also compatible with HotpotQA-style retrieval data after preprocessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class ListwiseExample:
    query: str
    documents: List[str]
    labels: List[int]  # 1 for positive, 0 for negatives


class ListwiseRerankerDataset(Dataset):
    """Reads JSONL listwise data and yields tokenized batches."""

    def __init__(
        self,
        path: str | Path,
        tokenizer_name: str,
        n_negatives: int = 7,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.examples: List[ListwiseExample] = list(self._load(path, n_negatives))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.n_negatives = n_negatives

    @staticmethod
    def _load(path: str | Path, n_neg: int) -> Iterable[ListwiseExample]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pos = rec["positive"]
                neg = list(rec.get("negatives", []))[:n_neg]
                # Pad short negative lists so every example has exactly n_neg
                if len(neg) < n_neg:
                    if len(neg) == 0:
                        neg = [pos] * n_neg
                    else:
                        i = 0
                        while len(neg) < n_neg:
                            neg.append(neg[i % len(neg)])
                            i += 1
                docs = [pos, *neg]
                labels = [1] + [0] * len(neg)
                yield ListwiseExample(rec["query"], docs, labels)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        pairs = [[ex.query, d] for d in ex.documents]
        enc = self.tokenizer(
            pairs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"],            # (N, L)
            "attention_mask": enc["attention_mask"],  # (N, L)
            "labels": torch.tensor(ex.labels, dtype=torch.float),
        }


def collate_listwise(batch: List[dict]) -> dict:
    """Stack ``(N, L)`` tensors into ``(B, N, L)`` for listwise training."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
