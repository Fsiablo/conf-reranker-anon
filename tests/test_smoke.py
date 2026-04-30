"""Smoke tests: import paths, loss math, selector logic.

These tests do NOT load any HuggingFace model — they only check the
non-encoder pieces and paper-level selection utilities.
"""

from __future__ import annotations

import math

import torch

from conf_reranker.inference import RiskBudgetConfig, conformal_threshold, risk_budgeted_topk
from conf_reranker.loss import ConfRerankerLoss, LossConfig


def test_import_top_level() -> None:
    import conf_reranker as pkg

    assert hasattr(pkg, "ConfReranker")
    assert hasattr(pkg, "ConfRerankerLoss")
    assert hasattr(pkg, "risk_budgeted_topk")


def test_loss_finite_on_random_input() -> None:
    torch.manual_seed(0)
    s = torch.randn(2, 5, requires_grad=True)
    c = torch.sigmoid(torch.randn(2, 5, requires_grad=True))
    y = torch.zeros(2, 5)
    y[:, 0] = 1.0  # first candidate is positive
    fn = ConfRerankerLoss(LossConfig())
    out = fn(s, c, y)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_risk_budgeted_topk_basic() -> None:
    # Two clearly confident candidates, three uncertain ones.
    s = torch.tensor([3.0, 2.5, 0.1, -0.2, -1.0])
    c = torch.tensor([0.95, 0.90, 0.40, 0.45, 0.30])
    sel, u, _, flag = risk_budgeted_topk(s, c, RiskBudgetConfig(rho=0.2))
    assert sel.numel() >= 1
    assert not flag  # the top two confident docs satisfy mean(c) >= 0.8
    assert torch.all(u >= 0)


def test_risk_budgeted_topk_low_conf_flag() -> None:
    # All low-confidence: no k satisfies budget → abstain with a flag.
    s = torch.tensor([1.0, 0.5, 0.0])
    c = torch.tensor([0.30, 0.25, 0.20])
    sel, _, _, flag = risk_budgeted_topk(s, c, RiskBudgetConfig(rho=0.1))
    assert flag is True
    assert sel.numel() == 0


def test_conformal_threshold_uses_irrelevant_upper_quantile() -> None:
    conf = [0.10, 0.20, 0.40, 0.70, 0.90]
    correct = [1, 0, 0, 1, 0]
    tau = conformal_threshold(conf, correct, alpha=0.34)
    assert math.isclose(tau, 0.90)


def test_entropy_regularizer_sign() -> None:
    fn = ConfRerankerLoss(LossConfig())
    c_uniform = torch.full((1, 4), 0.5)
    c_polar = torch.tensor([[0.99, 0.01, 0.99, 0.01]])
    # Entropy is HIGHER at 0.5; reg loss = -entropy is LOWER at 0.5.
    assert fn.reg_loss(c_uniform).item() < fn.reg_loss(c_polar).item()
    # And specifically, at p=0.5 reg_loss per element = -ln 2.
    assert math.isclose(
        fn.reg_loss(c_uniform).item(), -4 * math.log(2), rel_tol=1e-5
    )
