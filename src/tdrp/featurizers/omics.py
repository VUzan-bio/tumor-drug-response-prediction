from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def apply_pca(expr: pd.DataFrame, n_components: int = 256) -> Tuple[pd.DataFrame, PCA]:
    model = PCA(n_components=n_components, random_state=0)
    latent = model.fit_transform(expr.values)
    cols = [f"PC{i+1}" for i in range(latent.shape[1])]
    latent_df = pd.DataFrame(latent, index=expr.index, columns=cols)
    logger.info("PCA reduced %d genes to %d components", expr.shape[1], latent.shape[1])
    return latent_df, model


def transform_with_pca(expr: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    latent = pca.transform(expr.values)
    cols = [f"PC{i+1}" for i in range(latent.shape[1])]
    return pd.DataFrame(latent, index=expr.index, columns=cols)


def landmark_subset(expr: pd.DataFrame, landmark_genes: Optional[pd.Index]) -> pd.DataFrame:
    if landmark_genes is None:
        return expr
    missing = set(landmark_genes) - set(expr.columns)
    if missing:
        logger.warning("Missing %d landmark genes in expression matrix", len(missing))
    common = [g for g in landmark_genes if g in expr.columns]
    return expr[common]
