"""CLI entrypoint for training. Loads a YAML config and dispatches to trainer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from conf_reranker.loss import LossConfig
from conf_reranker.trainer import TrainConfig, train


def _build_cfg(d: dict) -> TrainConfig:
    loss_d = d.pop("loss", {}) or {}
    d.pop("inference", None)  # consumed by evaluator/selector, not trainer
    return TrainConfig(loss=LossConfig(**loss_d), **d)


def main() -> None:
    p = argparse.ArgumentParser(description="Train Conf-Reranker")
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with args.config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _build_cfg(raw)
    train(cfg)


if __name__ == "__main__":
    main()
