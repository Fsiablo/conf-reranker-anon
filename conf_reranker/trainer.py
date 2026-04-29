"""Minimal training loop for Conf-Reranker.

This is intentionally lean: a single-GPU PyTorch loop with mixed
precision, AdamW, linear-warmup-then-decay schedule, and gradient
accumulation. Distributed training, logging, and checkpoint averaging
are part of the planned release (see top-level README).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import ListwiseRerankerDataset, collate_listwise
from .loss import ConfRerankerLoss, LossConfig
from .model import ConfReranker, ConfRerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_path: str = "data/train.jsonl"
    output_dir: str = "runs/default"
    backbone_name: str = "BAAI/bge-reranker-v2-m3"
    n_negatives: int = 7
    max_length: int = 512
    batch_size: int = 4              # listwise groups per step
    grad_accum_steps: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    fp16: bool = True
    log_every: int = 50
    save_every: int = 1000
    seed: int = 42
    loss: LossConfig = field(default_factory=LossConfig)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", cfg.train_path)
    ds = ListwiseRerankerDataset(
        path=cfg.train_path,
        tokenizer_name=cfg.backbone_name,
        n_negatives=cfg.n_negatives,
        max_length=cfg.max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_listwise,
        num_workers=2,
        pin_memory=True,
    )

    logger.info("Building model from backbone %s", cfg.backbone_name)
    model = ConfReranker(ConfRerankerConfig(backbone_name=cfg.backbone_name)).to(device)
    loss_fn = ConfRerankerLoss(cfg.loss)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optim = torch.optim.AdamW(params, lr=cfg.lr, fused=True)

    total_steps = len(loader) * cfg.num_epochs // cfg.grad_accum_steps
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device == "cuda")

    step = 0
    model.train()
    for epoch in range(cfg.num_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)        # (B, N, L)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)              # (B, N)
            B, N, L = input_ids.shape
            input_ids = input_ids.view(B * N, L)
            attn = attn.view(B * N, L)

            with torch.amp.autocast("cuda", enabled=cfg.fp16 and device == "cuda"):
                s, c = model(input_ids, attention_mask=attn)
                s = s.view(B, N)
                c = c.view(B, N)
                out = loss_fn(s, c, labels)
                loss = out["loss"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.4f main=%.4f conf=%.4f reg=%.4f",
                    epoch, step, out["loss"].item(),
                    out["loss_main"].item(), out["loss_conf"].item(), out["loss_reg"].item(),
                )

            if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                ckpt = Path(cfg.output_dir) / f"step-{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt)

            step += 1

    final = Path(cfg.output_dir) / "final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final)
    logger.info("Training complete; checkpoint saved to %s", final)
