from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_variable_genes(expr: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    variances = expr.var(axis=0)
    top_genes = variances.sort_values(ascending=False).head(n_genes).index
    logger.info("Selected %d most variable genes", len(top_genes))
    return expr.loc[:, top_genes]


def zscore_per_gene(expr: pd.DataFrame) -> pd.DataFrame:
    means = expr.mean(axis=0)
    stds = expr.std(axis=0).replace(0, np.nan)
    standardized = (expr - means) / stds
    standardized = standardized.fillna(0.0)
    logger.info("Applied z-score normalization per gene")
    return standardized


def build_omics_matrix(expr: pd.DataFrame, cell_line_col: str = "cell_line") -> pd.DataFrame:
    expr = expr.set_index(cell_line_col)
    expr = expr.loc[~expr.index.duplicated(keep="first")]
    return expr


def compute_fingerprints(
    smiles: Iterable[str],
    radius: int,
    n_bits: int,
    ids: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Index]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("rdkit is required for fingerprint computation") from exc

    fps: list[np.ndarray] = []
    smiles_list = list(smiles)
    if ids is None:
        ids_list = [str(i) for i in range(len(smiles_list))]
    else:
        ids_list = list(ids)
        if len(ids_list) != len(smiles_list):
            raise ValueError("Number of SMILES does not match number of ids.")
    valid_ids: list[str] = []
    for entry, key in zip(smiles_list, ids_list):
        mol = Chem.MolFromSmiles(entry)
        if mol is None:
            continue
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(bit_vect, arr)
        fps.append(arr)
        valid_ids.append(str(key))
    frame = pd.DataFrame(fps, index=valid_ids)
    frame.index.name = "drug"
    return frame, frame.index


def align_labels(
    labels: pd.DataFrame,
    omics: pd.DataFrame,
    drugs: pd.DataFrame,
    cell_line_col: str = "cell_line",
    drug_col: str = "drug",
    target_col: str = "ln_ic50",
) -> pd.DataFrame:
    labels = labels.rename(columns={cell_line_col: "cell_line", drug_col: "drug", target_col: "target"})
    before = len(labels)
    labels = labels[labels["cell_line"].isin(omics.index) & labels["drug"].isin(drugs.index)]
    logger.info("Aligned labels: kept %d of %d pairs", len(labels), before)
    return labels.reset_index(drop=True)
