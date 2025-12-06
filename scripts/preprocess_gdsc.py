from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import pandas as pd

from tdrp.data.preprocess import align_labels, build_omics_matrix, select_variable_genes, zscore_per_gene
from tdrp.featurizers.drugs import smiles_to_fingerprints
from tdrp.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GDSC/CCLE drug response data.")
    parser.add_argument("--expression", type=pathlib.Path, required=True, help="Path to RNA-seq expression table (csv/parquet).")
    parser.add_argument("--labels", type=pathlib.Path, required=True, help="Path to drug response labels (cell_line, drug, ln_ic50).")
    parser.add_argument("--drug-smiles", type=pathlib.Path, required=True, help="Path to table with columns [drug, smiles].")
    parser.add_argument("--metadata", type=pathlib.Path, required=False, help="Optional metadata with cell_line,tissue columns.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data/processed"))
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--fingerprint-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def load_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    expr = load_table(args.expression)
    labels = load_table(args.labels)
    smiles = load_table(args.drug_smiles)

    expr = select_variable_genes(expr, n_genes=args.n_genes)
    expr = zscore_per_gene(expr)
    omics = build_omics_matrix(expr)

    fp_df = smiles_to_fingerprints(
        smiles=smiles.set_index("drug")["smiles"],
        radius=args.fingerprint_radius,
        n_bits=args.fingerprint_bits,
    )

    aligned = align_labels(labels, omics, fp_df)

    omics.to_parquet(args.outdir / "omics.parquet")
    fp_df.to_parquet(args.outdir / "drug_fingerprints.parquet")
    aligned.to_parquet(args.outdir / "labels.parquet")

    if args.metadata:
        metadata = load_table(args.metadata)
        metadata.to_parquet(args.outdir / "metadata.parquet")


if __name__ == "__main__":
    main()
