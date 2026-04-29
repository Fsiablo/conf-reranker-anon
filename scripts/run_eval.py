"""CLI entrypoint for evaluation.

Computes MRR, nDCG@k, Recall@k, ECE, and reports the average selected
set size and low-confidence flag rate under the configured risk
budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from conf_reranker.data import ListwiseRerankerDataset, collate_listwise
from conf_reranker.inference import RiskBudgetConfig, risk_budgeted_topk
from conf_reranker.model import ConfReranker, ConfRerankerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(scored: List[int], k: int) -> float:
    top_k = scored[:k]
    ideal = sorted(scored, reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(top_k) / idcg if idcg > 0 else 0.0


def mrr(scored: List[int]) -> float:
    for i, r in enumerate(scored):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(scored: List[int], k: int) -> float:
    return float(sum(scored[:k]) > 0)


def ece(probs: List[float], correct: List[int], n_bins: int = 15) -> float:
    if not probs:
        return 0.0
    e = 0.0
    n = len(probs)
    edges = [i / n_bins for i in range(n_bins + 1)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_idx = [i for i, p in enumerate(probs) if lo < p <= hi or (lo == 0 and p == 0)]
        if not bin_idx:
            continue
        avg_conf = sum(probs[i] for i in bin_idx) / len(bin_idx)
        avg_acc = sum(correct[i] for i in bin_idx) / len(bin_idx)
        e += (len(bin_idx) / n) * abs(avg_conf - avg_acc)
    return e


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(ckpt_path: Path, data_path: Path, backbone: str, rho: float) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConfReranker(ConfRerankerConfig(backbone_name=backbone)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    ds = ListwiseRerankerDataset(data_path, backbone)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_listwise)

    metrics = {"mrr": 0.0, "ndcg@5": 0.0, "recall@1": 0.0, "recall@5": 0.0}
    all_conf, all_correct = [], []
    n_low_flag = 0
    total_k = 0
    n = 0
    sel_cfg = RiskBudgetConfig(rho=rho)

    for batch in loader:
        ids = batch["input_ids"].squeeze(0).to(device)
        msk = batch["attention_mask"].squeeze(0).to(device)
        labels = batch["labels"].squeeze(0).cpu().tolist()
        s, c = model(ids, attention_mask=msk)
        order = torch.argsort(s, descending=True).cpu().tolist()
        scored = [int(labels[i]) for i in order]

        metrics["mrr"] += mrr(scored)
        metrics["ndcg@5"] += ndcg_at_k(scored, 5)
        metrics["recall@1"] += recall_at_k(scored, 1)
        metrics["recall@5"] += recall_at_k(scored, 5)

        all_conf.extend(c.cpu().tolist())
        all_correct.extend([int(x) for x in labels])

        sel, _, _, flag = risk_budgeted_topk(s, c, sel_cfg)
        total_k += sel.numel()
        n_low_flag += int(flag)
        n += 1

    for k in metrics:
        metrics[k] /= max(n, 1)
    metrics["ece"] = ece(all_conf, all_correct)
    metrics["avg_kstar"] = total_k / max(n, 1)
    metrics["low_conf_flag_rate"] = n_low_flag / max(n, 1)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Conf-Reranker")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--backbone", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--rho", type=float, default=0.2)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    metrics = evaluate(args.ckpt, args.data, args.backbone, args.rho)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
