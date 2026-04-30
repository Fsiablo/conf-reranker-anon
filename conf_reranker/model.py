"""Dual-head cross-encoder for confidence-aware reranking.

Implements the architecture in Section III-A of the paper: a shared
encoder feeds two parallel two-layer MLP heads — a scoring head
producing logit s_i, and a confidence head producing c_i in (0, 1)
via sigmoid. Stop-gradient is applied at the encoder boundary on the
confidence-head input so that the encoder is updated *only* by the
ranking signal (asymmetric co-training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class ConfRerankerConfig:
    backbone_name: str = "BAAI/bge-reranker-v2-m3"
    hidden_dropout: float = 0.1
    head_hidden_ratio: float = 0.25  # paper default: bottleneck hidden dim = d / 4
    pooling: str = "cls"  # "cls" | "mean"


class _MLPHead(nn.Module):
    """Two-layer MLP head, tanh activation, single-scalar output (Eq. 3)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(h)))).squeeze(-1)


class ConfReranker(nn.Module):
    """Dual-head cross-encoder reranker.

    Forward returns ``(s, c)`` where ``s`` are raw ranking logits and
    ``c`` are calibrated confidences in (0, 1). The confidence head
    sees a stop-gradient'd encoder representation, implementing the
    asymmetric coupling described in Eq. (1)-(2) of the paper.
    """

    def __init__(self, config: Optional[ConfRerankerConfig] = None) -> None:
        super().__init__()
        self.config = config or ConfRerankerConfig()
        import transformers as _tf
        _enc_kwargs = {}
        if tuple(int(x) for x in _tf.__version__.split(".")[:2]) >= (4, 36):
            _enc_kwargs["attn_implementation"] = "sdpa"
        self.encoder = AutoModel.from_pretrained(
            self.config.backbone_name, **_enc_kwargs
        )
        d = self.encoder.config.hidden_size
        h = max(1, int(d * self.config.head_hidden_ratio))
        self.score_head = _MLPHead(d, h, self.config.hidden_dropout)
        self.conf_head = _MLPHead(d, h, self.config.hidden_dropout)

    def _pool(self, out, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.config.pooling == "cls":
            return out.last_hidden_state[:, 0]
        h = out.last_hidden_state
        if mask is None:
            return h.mean(dim=1)
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        h = self._pool(out, attention_mask)

        # Asymmetric stop-gradient (Eq. 2): conf head sees detached h
        s = self.score_head(h)
        c = torch.sigmoid(self.conf_head(h.detach()))
        return s, c

    @torch.no_grad()
    def score(
        self,
        query: str,
        documents: list[str],
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a single (query, [docs]) pair end-to-end. Useful for demo."""
        tok = tokenizer or AutoTokenizer.from_pretrained(self.config.backbone_name)
        device = device or next(self.parameters()).device
        pairs = [[query, d] for d in documents]
        enc = tok(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device)
        return self.forward(**enc)
