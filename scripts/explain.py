from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tdrp.analysis.interpretability import compute_shap_values
from tdrp.config import Config, default_config, load_config
from tdrp.data.dataloaders import DrugResponseDataset, Example
from tdrp.models.fusion import MultiModalRegressor
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for trained model.")
    parser.add_argument("--config", type=pathlib.Path, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("outputs/model.pt"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs/shap_values.npz"))
    parser.add_argument("--max-eval", type=int, default=256)
    return parser.parse_args()


def build_dataset(omics: pd.DataFrame, drugs: pd.DataFrame, labels: pd.DataFrame):
    examples = [Example(cell_line=row["cell_line"], drug=row["drug"], target=float(row["target"])) for _, row in labels.iterrows()]
    return DrugResponseDataset(omics=omics, drugs=drugs, examples=examples)


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg: Config = load_config(args.config) if args.config else default_config()

    omics = pd.read_parquet(cfg.data.omics_path)
    drugs = pd.read_parquet(cfg.data.drugs_path)
    labels = pd.read_parquet(cfg.data.labels_path)

    dataset = build_dataset(omics, drugs, labels)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    model = MultiModalRegressor(
        omics_dim=omics.shape[1],
        drug_dim=drugs.shape[1],
        omics_hidden=cfg.model.hidden_omics,
        omics_latent=cfg.model.omics_latent_dim,
        drug_latent=cfg.model.drug_latent_dim,
        fusion_hidden=cfg.model.fusion_hidden,
        dropout=cfg.model.dropout,
        use_vae=cfg.model.use_vae,
        vae_latent=cfg.model.vae_latent_dim,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    feature_names = list(omics.columns) + [f"fp_{i}" for i in range(drugs.shape[1])]
    shap_values = compute_shap_values(
        model=model,
        loader=loader,
        omics_dim=omics.shape[1],
        drug_dim=drugs.shape[1],
        device=device,
        max_eval=args.max_eval,
        feature_names=feature_names,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, values=shap_values.values, data=shap_values.data, feature_names=np.array(feature_names))
    print(f"Saved SHAP explanation to {args.output}")


if __name__ == "__main__":
    main()
