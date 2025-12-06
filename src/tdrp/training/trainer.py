from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..models.losses import RegressionLoss
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    best_val: float = float("inf")
    patience_counter: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        vae_beta: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip = gradient_clip
        self.loss_fn = RegressionLoss(vae_beta=vae_beta)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Tuple[float, float]:
        mode = "train" if train else "eval"
        self.model.train(train)
        total_loss = 0.0
        total_mse = 0.0
        count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                omics = batch["omics"].to(self.device)
                drug = batch["drug"].to(self.device)
                target = batch["target"].to(self.device)
                if train:
                    self.optimizer.zero_grad()
                pred, aux = self.model(omics, drug)
                losses = self.loss_fn(pred, target, aux, omics)
                loss = losses["total"]
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                total_loss += loss.item() * len(target)
                total_mse += losses["mse"].item() * len(target)
                count += len(target)
        return total_loss / count, total_mse / count

    def fit(
        self,
        loaders: Dict[str, DataLoader],
        max_epochs: int = 200,
        patience: int = 20,
    ) -> TrainState:
        state = TrainState()
        for epoch in range(max_epochs):
            state.epoch = epoch
            train_loss, _ = self._run_epoch(loaders["train"], train=True)
            val_loss, _ = self._run_epoch(loaders.get("val", loaders["train"]), train=False)
            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            state.history.append(metrics)
            logger.info("Epoch %d - train %.4f - val %.4f", epoch, train_loss, val_loss)
            if val_loss < state.best_val:
                state.best_val = val_loss
                state.patience_counter = 0
                state.best_state = self._snapshot()
            else:
                state.patience_counter += 1
            if state.patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break
        if hasattr(state, "best_state"):
            self.model.load_state_dict(state.best_state)  # type: ignore[arg-type]
        return state

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                omics = batch["omics"].to(self.device)
                drug = batch["drug"].to(self.device)
                target = batch["target"].to(self.device)
                pred, _ = self.model(omics, drug)
                preds.append(pred)
                targets.append(target)
        pred_tensor = torch.cat(preds, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        return compute_metrics(pred_tensor, target_tensor)

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
