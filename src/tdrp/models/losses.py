from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class RegressionLoss(nn.Module):
    def __init__(self, vae_beta: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.beta = vae_beta

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        omics_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {"mse": self.mse(pred, target)}
        if aux.get("reconstruction") is not None:
            recon = aux["reconstruction"]
            mu = aux["mu"]
            logvar = aux["logvar"]
            assert recon is not None and mu is not None and logvar is not None
            recon_loss = self.mse(recon, omics_input)
            kl = self._kl_divergence(mu, logvar)
            losses["recon"] = recon_loss
            losses["kl"] = kl
            losses["total"] = losses["mse"] + recon_loss + self.beta * kl
        else:
            losses["total"] = losses["mse"]
        return losses
