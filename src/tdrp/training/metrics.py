from __future__ import annotations

from typing import Dict

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def pearsonr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    if denominator.item() == 0:
        return torch.tensor(0.0, device=pred.device)
    return numerator / denominator


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "rmse": float(rmse(pred, target).item()),
            "pearson": float(pearsonr(pred, target).item()),
        }
