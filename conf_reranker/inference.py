"""Risk-Budgeted Top-k* inference (Algorithm 1 of the paper).

Given per-candidate (s_i, c_i) outputs, this module performs:
    1. Confidence-dependent temperature scaling (Eq. 8):
            s'_i = s_i * (c_i + eps) / T_0
    2. Softmax over s' to obtain p_i.
    3. Utility composition (Eq. 9):  u_i = p_i * c_i^beta.
    4. Risk-budgeted set selection (Eq. 10):
            k* = min { k : avg(c_(1..k)) >= 1 - rho }.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RiskBudgetConfig:
    T0: float = 0.8
    beta: float = 2.0
    rho: float = 0.2          # risk budget; k* requires mean conf >= 1 - rho
    eps: float = 1e-6
    k_min: int = 1
    k_max: Optional[int] = None  # default: N


def risk_budgeted_topk(
    s: torch.Tensor,
    c: torch.Tensor,
    cfg: Optional[RiskBudgetConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply Algorithm 1 to a single ranking list.

    Args:
        s: (N,) raw ranking logits.
        c: (N,) confidence scores in (0, 1).
        cfg: RiskBudgetConfig.

    Returns:
        selected_idx : LongTensor (k*,) — indices of the chosen set,
                       sorted by utility descending.
        u            : Tensor (N,)   — full utility vector.
        c_sorted     : Tensor (N,)   — confidence aligned to utility order.
        flag_low_conf: bool          — True if no k satisfied the budget,
                                       in which case the full set is returned
                                       and downstream code SHOULD escalate.
    """
    cfg = cfg or RiskBudgetConfig()
    assert s.dim() == 1 and c.dim() == 1 and s.shape == c.shape, "s and c must be (N,)"
    N = s.shape[0]
    k_max = cfg.k_max if cfg.k_max is not None else N
    k_max = min(k_max, N)

    s_prime = s * (c + cfg.eps) / cfg.T0
    p = F.softmax(s_prime, dim=-1)
    u = p * (c.clamp(min=cfg.eps) ** cfg.beta)

    order = torch.argsort(u, descending=True)
    c_sorted = c[order]
    cum_mean = torch.cumsum(c_sorted, dim=0) / torch.arange(
        1, N + 1, device=c.device, dtype=c.dtype
    )

    threshold = 1.0 - cfg.rho
    satisfied = cum_mean >= threshold
    # enforce k_min..k_max window
    valid = satisfied.clone()
    if cfg.k_min > 1:
        valid[: cfg.k_min - 1] = False
    if k_max < N:
        valid[k_max:] = False

    if valid.any():
        k_star = int(torch.nonzero(valid, as_tuple=False)[0].item()) + 1
        return order[:k_star], u, c_sorted, False
    # no k satisfies budget → return full set + low-confidence flag
    return order[:k_max], u, c_sorted, True


class RiskBudgetedSelector:
    """Stateful wrapper around :func:`risk_budgeted_topk` for batch use."""

    def __init__(self, cfg: Optional[RiskBudgetConfig] = None) -> None:
        self.cfg = cfg or RiskBudgetConfig()

    def __call__(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
    ) -> List[dict]:
        """Apply per-row to a (B, N) batch.

        Returns a list of length B; each entry is a dict with keys
        ``selected_idx``, ``utility``, ``low_conf_flag``.
        """
        if s.dim() == 1:
            s, c = s.unsqueeze(0), c.unsqueeze(0)
        out = []
        for i in range(s.shape[0]):
            sel, u, _, flag = risk_budgeted_topk(s[i], c[i], self.cfg)
            out.append(
                {
                    "selected_idx": sel.tolist(),
                    "utility": u.detach().cpu().tolist(),
                    "low_conf_flag": flag,
                }
            )
        return out


def conformal_threshold(
    cal_confidences: Sequence[float],
    cal_correctness: Sequence[int],
    alpha: float = 0.1,
) -> float:
    """Split-conformal calibration of the confidence threshold.

    Companion to Theorem 2 (Section V-C). Given calibration set
    confidences and binary correctness labels, returns the threshold
    tau such that with probability >= 1 - alpha the held-out
    expected precision exceeds the target.

    This is intentionally a minimal placeholder and not the full
    risk-control variant of the paper.
    """
    import numpy as np

    c = np.asarray(cal_confidences, dtype=float)
    y = np.asarray(cal_correctness, dtype=int)
    if c.size == 0:
        return 0.0
    # nonconformity score: 1 - c on correct, c on incorrect
    scores = np.where(y == 1, 1.0 - c, c)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = float(np.clip(q_level, 0.0, 1.0))
    return float(np.quantile(scores, q_level, method="higher"))
