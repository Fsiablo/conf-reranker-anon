"""Ranker--Auditor co-training objective.

Implements Eq. (4)-(7) of the paper:

    L = L_main + lambda_c * L_conf + lambda_r * L_reg

with
    L_main : confidence-weighted listwise softmax cross-entropy
    L_conf : self-consistency between c_i and sigmoid(s_i)
    L_reg  : negative binary entropy of c_i (penalizes collapse)

All stop-gradient operations follow the asymmetric scheme: the auditor
weights propagate to the ranker only as scalar multipliers; the
self-consistency term drives only the confidence head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    lambda_c: float = 0.9       # weight of self-consistency term
    lambda_r: float = 0.2       # weight of entropy regularizer
    eps: float = 1e-7           # numerical floor for log


class ConfRerankerLoss(nn.Module):
    """Combined three-term objective (Eq. 4 in the paper)."""

    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.cfg = config or LossConfig()

    @staticmethod
    def main_loss(s: torch.Tensor, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Confidence-weighted listwise softmax cross-entropy (Eq. 5).

        Shapes: s, c, y are ``(B, N)`` per listwise group.
        """
        log_p = F.log_softmax(s, dim=-1)
        # stop-gradient on c: auditor provides multiplicative weights only
        w = c.detach()
        per_sample = -(w * y * log_p).sum(dim=-1)
        return per_sample.mean()

    def conf_loss(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Self-consistency: pull c toward sigmoid(s) (Eq. 6)."""
        target = torch.sigmoid(s).detach()
        return F.mse_loss(c, target)

    def reg_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Negative binary entropy of ``c`` (Eq. 13).

        Combined with :meth:`conf_loss` this prevents the auditor from
        collapsing to a constant value.
        """
        c = c.clamp(self.cfg.eps, 1.0 - self.cfg.eps)
        h = c * torch.log(c) + (1.0 - c) * torch.log(1.0 - c)
        return h.sum(dim=-1).mean()

    def forward(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        l_main = self.main_loss(s, c, y)
        l_conf = self.conf_loss(s, c)
        l_reg = self.reg_loss(c)
        total = l_main + self.cfg.lambda_c * l_conf + self.cfg.lambda_r * l_reg
        return {
            "loss": total,
            "loss_main": l_main.detach(),
            "loss_conf": l_conf.detach(),
            "loss_reg": l_reg.detach(),
        }
