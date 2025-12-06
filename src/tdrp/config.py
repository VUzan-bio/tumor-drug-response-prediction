from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, Optional

import yaml


@dataclasses.dataclass
class DataConfig:
    omics_path: pathlib.Path
    drugs_path: pathlib.Path
    labels_path: pathlib.Path
    metadata_path: Optional[pathlib.Path] = None
    gene_subset: str = "top_variance"
    n_genes: int = 2000
    fingerprint_radius: int = 2
    fingerprint_bits: int = 1024
    cache_dir: pathlib.Path = pathlib.Path("data/processed")


@dataclasses.dataclass
class ModelConfig:
    omics_encoder: str = "mlp"
    omics_latent_dim: int = 256
    drug_latent_dim: int = 256
    hidden_omics: int = 512
    hidden_drug: int = 256
    fusion_hidden: int = 128
    dropout: float = 0.2
    use_vae: bool = False
    vae_latent_dim: int = 64
    vae_beta: float = 1.0


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    gradient_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    target: str = "ln_ic50"
    split_strategy: str = "leave_cell_line_out"
    tissue_holdout: Optional[str] = None
    seed: int = 42


@dataclasses.dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _path_from_config(value: Any) -> Any:
    if isinstance(value, str):
        return pathlib.Path(value)
    if isinstance(value, dict):
        return {k: _path_from_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_path_from_config(v) for v in value]
    return value


def load_config(path: pathlib.Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)
    raw = _path_from_config(raw)
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    return Config(data=data, model=model, training=training)


def default_config() -> Config:
    return Config(
        data=DataConfig(
            omics_path=pathlib.Path("data/processed/omics.parquet"),
            drugs_path=pathlib.Path("data/processed/drug_fingerprints.parquet"),
            labels_path=pathlib.Path("data/processed/labels.parquet"),
            metadata_path=pathlib.Path("data/processed/metadata.parquet"),
        ),
        model=ModelConfig(),
        training=TrainingConfig(),
    )
