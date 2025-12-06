from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def leave_cell_line_out(labels: pd.DataFrame, test_cell_lines: Iterable[str]) -> Dict[str, List[int]]:
    test_set = set(test_cell_lines)
    idx = np.arange(len(labels))
    mask = labels["cell_line"].isin(test_set)
    return {
        "train": idx[~mask].tolist(),
        "val": idx[~mask].tolist(),
        "test": idx[mask].tolist(),
    }


def tissue_holdout(labels: pd.DataFrame, metadata: pd.DataFrame, holdout_tissue: str) -> Dict[str, List[int]]:
    tissues = metadata.set_index("cell_line")["tissue"]
    labels = labels.copy()
    labels["tissue"] = labels["cell_line"].map(tissues)
    idx = np.arange(len(labels))
    holdout_mask = labels["tissue"] == holdout_tissue
    return {
        "train": idx[~holdout_mask].tolist(),
        "val": idx[~holdout_mask].tolist(),
        "test": idx[holdout_mask].tolist(),
    }


def kfold_cell_line_split(labels: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Iterable[Dict[str, List[int]]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cell_lines = labels["cell_line"].unique()
    for fold, (train_idx, test_idx) in enumerate(kf.split(cell_lines)):
        train_cells = set(cell_lines[train_idx])
        test_cells = set(cell_lines[test_idx])
        mask_train = labels["cell_line"].isin(train_cells)
        mask_test = labels["cell_line"].isin(test_cells)
        idx = np.arange(len(labels))
        split = {
            "train": idx[mask_train].tolist(),
            "val": idx[mask_train].tolist(),
            "test": idx[mask_test].tolist(),
        }
        logger.info("Built fold %d with %d train and %d test pairs", fold, len(split["train"]), len(split["test"]))
        yield split
