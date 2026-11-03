"""Composition scoring and the brief's metrics (3.3): Top-1/Top-5, harmonic mean, confusion matrix."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def comp_scores(crop_logits: torch.Tensor, disease_logits: torch.Tensor, cd: torch.Tensor) -> torch.Tensor:
    """Score all crop-disease compositions: log P(crop) + log P(disease).  Returns (B, num_comps).
    Works for unseen comps because each one is built from a seen crop and a seen disease."""
    lc = F.log_softmax(crop_logits.float(), dim=1)
    ld = F.log_softmax(disease_logits.float(), dim=1)
    return lc[:, cd[:, 0]] + ld[:, cd[:, 1]]


def topk_correct(scores: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    top = scores.topk(max(ks), dim=1).indices
    hit = top.eq(target.unsqueeze(1))
    return {k: int(hit[:, :k].any(dim=1).sum()) for k in ks}


def harmonic_mean(seen: float, unseen: float) -> float:
    return 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
