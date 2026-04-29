"""End-to-end CPU demo for Conf-Reranker.

Loads a tiny encoder backbone, scores a synthetic (query, [docs])
list, and applies risk-budgeted Top-k* selection. Designed to run in
under a minute on a laptop CPU and to verify that the package
imports and forward / inference paths work without GPUs or large
checkpoints.

Usage:
    python -m scripts.demo
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from conf_reranker.inference import RiskBudgetConfig, risk_budgeted_topk
from conf_reranker.model import ConfReranker, ConfRerankerConfig


SMALL_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"  # ~22M params, CPU-friendly


def main() -> None:
    print("[demo] loading model:", SMALL_BACKBONE)
    cfg = ConfRerankerConfig(backbone_name=SMALL_BACKBONE)
    model = ConfReranker(cfg).eval()
    tok = AutoTokenizer.from_pretrained(SMALL_BACKBONE)

    query = "How do I set a clock period in OpenROAD?"
    docs = [
        "create_clock -name clk -period 2.0 [get_ports clk]",
        "OpenROAD reads SDC files via read_sdc <file>.",
        "DEF stands for Design Exchange Format.",
        "Place_design performs detailed placement after global placement.",
        "Route_design runs detailed routing using TritonRoute.",
        "OpenROAD is an open-source RTL-to-GDSII flow under active development.",
    ]

    s, c = model.score(query, docs, tokenizer=tok)
    print("\n[demo] raw outputs (s, c):")
    for d, si, ci in zip(docs, s.tolist(), c.tolist()):
        print(f"  s={si:+.3f}  c={ci:.3f}  | {d}")

    sel, u, _, low = risk_budgeted_topk(s, c, RiskBudgetConfig(rho=0.2))
    print("\n[demo] risk-budgeted Top-k* selection (rho=0.2):")
    print(f"  k* = {sel.numel()}, low_conf_flag = {low}")
    for rank, idx in enumerate(sel.tolist(), 1):
        print(f"  #{rank}  u={u[idx]:.4f}  c={c[idx]:.3f}  | {docs[idx]}")

    print("\n[demo] OK — package import + forward + selection paths work.")
    print("[demo] NOTE: model is *un-trained*; values are illustrative only.")


if __name__ == "__main__":
    main()
