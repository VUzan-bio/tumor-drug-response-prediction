from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Example:
    cell_line: str
    drug: str
    target: float
    tissue: Optional[str] = None


class DrugResponseDataset(Dataset):
    def __init__(
        self,
        omics: pd.DataFrame,
        drugs: pd.DataFrame,
        examples: Sequence[Example],
    ) -> None:
        self.omics = omics
        self.drugs = drugs
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        omics_row = self.omics.loc[example.cell_line].astype(np.float32).values
        drug_row = self.drugs.loc[example.drug].astype(np.float32).values
        return {
            "omics": torch.from_numpy(omics_row),
            "drug": torch.from_numpy(drug_row),
            "target": torch.tensor(example.target, dtype=torch.float32),
            "tissue": example.tissue or "",
        }


def _build_examples(labels: pd.DataFrame, tissues: Optional[pd.Series]) -> List[Example]:
    examples: List[Example] = []
    for _, row in labels.iterrows():
        tissue = None
        if tissues is not None:
            tissue = tissues.get(row["cell_line"], None)
        examples.append(
            Example(
                cell_line=row["cell_line"],
                drug=row["drug"],
                target=float(row["target"]),
                tissue=tissue,
            )
        )
    return examples


def make_loaders(
    omics: pd.DataFrame,
    drugs: pd.DataFrame,
    labels: pd.DataFrame,
    tissues: Optional[pd.Series],
    splits: Dict[str, Iterable[int]],
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    labels = labels.reset_index(drop=True)
    for split_name, split_indices in splits.items():
        idx = list(split_indices)
        subset = labels.iloc[idx]
        examples = _build_examples(subset, tissues)
        ds = DrugResponseDataset(omics=omics, drugs=drugs, examples=examples)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
        )
    return loaders
