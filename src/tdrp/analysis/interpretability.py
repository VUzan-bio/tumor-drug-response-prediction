from __future__ import annotations

from typing import Iterable, List, Optional

import torch


def compute_shap_values(
    model: torch.nn.Module,
    loader,
    omics_dim: int,
    drug_dim: int,
    device: torch.device,
    background_size: int = 128,
    max_eval: int = 512,
    feature_names: Optional[List[str]] = None,
):
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError("shap package is required for interpretability") from exc

    model.eval()
    backgrounds: List[torch.Tensor] = []
    evals: List[torch.Tensor] = []
    for batch in loader:
        omics = batch["omics"]
        drug = batch["drug"]
        combined = torch.cat([omics, drug], dim=1)
        if len(backgrounds) * combined.shape[0] < background_size:
            backgrounds.append(combined)
        if len(evals) * combined.shape[0] < max_eval:
            evals.append(combined)
        if len(evals) * combined.shape[0] >= max_eval and len(backgrounds) * combined.shape[0] >= background_size:
            break
    background_tensor = torch.cat(backgrounds, dim=0)[:background_size].to(device)
    eval_tensor = torch.cat(evals, dim=0)[:max_eval].to(device)

    class Wrapped(torch.nn.Module):
        def __init__(self, base: torch.nn.Module):
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor):
            omics = x[:, :omics_dim]
            drug = x[:, omics_dim : omics_dim + drug_dim]
            pred, _ = self.base(omics, drug)
            return pred

    wrapped = Wrapped(model).to(device)
    explainer = shap.DeepExplainer(wrapped, background_tensor)
    values = explainer.shap_values(eval_tensor)
    if isinstance(values, list):
        values = values[0]
    return shap.Explanation(values=values, data=eval_tensor.cpu().numpy(), feature_names=feature_names)
