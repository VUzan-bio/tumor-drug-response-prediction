from __future__ import annotations

import argparse
import pathlib
import random

import pandas as pd
import torch

from tdrp.config import Config, default_config, load_config
from tdrp.data.dataloaders import make_loaders
from tdrp.data.splits import kfold_cell_line_split, leave_cell_line_out, tissue_holdout
from tdrp.models.fusion import MultiModalRegressor
from tdrp.training.trainer import Trainer
from tdrp.utils.logging import setup_logging
from tdrp.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-modal drug response model.")
    parser.add_argument("--config", type=pathlib.Path, help="Path to YAML config.")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def select_split(cfg: Config, labels: pd.DataFrame, metadata: pd.DataFrame):
    strategy = cfg.training.split_strategy
    if strategy == "tissue_holdout" and cfg.training.tissue_holdout:
        if metadata.empty:
            raise ValueError("tissue_holdout split requires metadata with tissue annotations.")
        return tissue_holdout(labels, metadata, cfg.training.tissue_holdout)
    if strategy == "leave_cell_line_out":
        cell_lines = sorted(labels["cell_line"].unique())
        random.seed(cfg.training.seed)
        random.shuffle(cell_lines)
        holdout = cell_lines[: max(1, int(0.2 * len(cell_lines)))]
        return leave_cell_line_out(labels, holdout)
    if strategy == "kfold":
        # Use first fold for simplicity; research experiments can iterate.
        return next(iter(kfold_cell_line_split(labels, n_splits=5, seed=cfg.training.seed)))
    raise ValueError(f"Unknown split strategy: {strategy}")


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    cfg = load_config(args.config) if args.config else default_config()
    set_seed(cfg.training.seed)

    omics = pd.read_parquet(cfg.data.omics_path)
    drugs = pd.read_parquet(cfg.data.drugs_path)
    labels = pd.read_parquet(cfg.data.labels_path)
    metadata = pd.read_parquet(cfg.data.metadata_path) if cfg.data.metadata_path and cfg.data.metadata_path.exists() else pd.DataFrame()
    tissues = metadata.set_index("cell_line")["tissue"] if not metadata.empty else None

    splits = select_split(cfg, labels, metadata)
    loaders = make_loaders(
        omics=omics,
        drugs=drugs,
        labels=labels,
        tissues=tissues,
        splits=splits,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    model = MultiModalRegressor(
        omics_dim=omics.shape[1],
        drug_dim=drugs.shape[1],
        omics_hidden=cfg.model.hidden_omics,
        omics_latent=cfg.model.omics_latent_dim,
        drug_latent=cfg.model.drug_latent_dim,
        drug_hidden=cfg.model.hidden_drug,
        fusion_hidden=cfg.model.fusion_hidden,
        dropout=cfg.model.dropout,
        use_vae=cfg.model.use_vae,
        vae_latent=cfg.model.vae_latent_dim,
    )
    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        gradient_clip=cfg.training.gradient_clip,
        vae_beta=cfg.model.vae_beta,
    )
    state = trainer.fit(loaders, max_epochs=cfg.training.max_epochs, patience=cfg.training.patience)
    metrics = trainer.evaluate(loaders["test"])

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.pt"
    torch.save(model.state_dict(), model_path)
    metrics_path = args.output / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for k, v in metrics.items():
            handle.write(f"{k}: {v:.4f}\n")

    print(f"Finished training at epoch {state.epoch}, val loss {state.best_val:.4f}")
    print(f"Test metrics: {metrics}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
