from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .encoders import DrugEncoder, OmicsEncoder, VariationalAutoencoder


class MultiModalRegressor(nn.Module):
    def __init__(
        self,
        omics_dim: int,
        drug_dim: int,
        omics_hidden: int = 512,
        omics_latent: int = 256,
        drug_latent: int = 256,
        drug_hidden: int = 256,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
        use_vae: bool = False,
        vae_latent: int = 64,
    ):
        super().__init__()
        self.use_vae = use_vae
        if use_vae:
            self.vae = VariationalAutoencoder(
                input_dim=omics_dim,
                latent_dim=vae_latent,
                hidden_dim=omics_hidden,
                dropout=dropout,
            )
            omics_latent_dim = vae_latent
        else:
            self.omics_encoder = OmicsEncoder(
                input_dim=omics_dim,
                hidden_dim=omics_hidden,
                latent_dim=omics_latent,
                dropout=dropout,
            )
            omics_latent_dim = omics_latent

        self.drug_encoder = DrugEncoder(
            input_dim=drug_dim,
            hidden_dim=drug_hidden,
            latent_dim=drug_latent,
            dropout=dropout,
        )
        fusion_dim = omics_latent_dim + drug_latent
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, omics: torch.Tensor, drug: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        aux: Dict[str, Optional[torch.Tensor]] = {"reconstruction": None, "mu": None, "logvar": None}
        if self.use_vae:
            recon, mu, logvar = self.vae(omics)
            aux["reconstruction"] = recon
            aux["mu"] = mu
            aux["logvar"] = logvar
            omics_latent = mu
        else:
            omics_latent = self.omics_encoder(omics)
        drug_latent = self.drug_encoder(drug)
        fusion = torch.cat([omics_latent, drug_latent], dim=1)
        pred = self.regressor(fusion).squeeze(1)
        return pred, aux
