from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def smiles_to_fingerprints(
    smiles: Iterable[str],
    radius: int = 2,
    n_bits: int = 1024,
    ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError("rdkit is required for fingerprint computation") from exc

    fps: list[np.ndarray] = []
    smiles_list = list(smiles) if not isinstance(smiles, pd.Series) else list(smiles.values)
    if ids is None:
        if isinstance(smiles, pd.Series):
            ids = list(smiles.index)
        else:
            ids = [str(i) for i in range(len(smiles_list))]
    ids_list = list(ids)
    if len(ids_list) != len(smiles_list):
        raise ValueError("Number of SMILES does not match number of ids.")

    valid_ids: list[str] = []
    for key, smi in zip(ids_list, smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Skipping invalid SMILES for %s", key)
            continue
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(bit_vect, arr)
        fps.append(arr)
        valid_ids.append(str(key))
    frame = pd.DataFrame(fps, index=valid_ids)
    frame.index.name = "drug"
    logger.info("Computed fingerprints for %d molecules", len(frame))
    return frame


def smiles_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit is required for graph featurization") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    adjacency = np.array(edges, dtype=np.int64)
    return atoms, adjacency
