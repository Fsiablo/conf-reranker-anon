"""Conf-Reranker: a confidence-propagating cross-encoder reranker for EDA RAG.

Reference paper:
    Anonymous Authors.
    "Conf-Reranker: Reliability-Aware Reranking for EDA-Oriented
    Retrieval-Augmented Generation."
    IEEE Transactions on Computer-Aided Design of Integrated Circuits
    and Systems (TCAD).
"""

from .model import ConfReranker
from .loss import ConfRerankerLoss
from .inference import risk_budgeted_topk, RiskBudgetedSelector

__version__ = "0.1.0"
__all__ = [
    "ConfReranker",
    "ConfRerankerLoss",
    "risk_budgeted_topk",
    "RiskBudgetedSelector",
]
