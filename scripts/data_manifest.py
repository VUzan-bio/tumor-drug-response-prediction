"""
Generate a JSON manifest summarizing processed tables and optional split CSVs.

Example:
python scripts/data_manifest.py \
  --processed-dir data/processed \
  --splits-dir data/processed/splits \
  --out outputs/data_manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.utils.io import load_parquet, save_json
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a data manifest from processed tables and splits.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with processed parquet files.")
    parser.add_argument("--splits-dir", default=None, help="Optional directory with split CSVs to summarize.")
    parser.add_argument("--out", default="outputs/data_manifest.json", help="Output JSON path.")
    return parser.parse_args()


def _summarize_processed(
    omics_df: pd.DataFrame,
    drug_fp_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict:
    omics_features = [c for c in omics_df.columns if c != "cell_line"]
    drug_features = [c for c in drug_fp_df.columns if c != "drug"]
    zero_fp = None
    if drug_features:
        zero_fp = float((drug_fp_df[drug_features].sum(axis=1) == 0).mean())
    label_stats = labels_df["ln_ic50"].astype(float).describe().to_dict()
    label_stats = {k: float(v) for k, v in label_stats.items() if k in {"min", "mean", "50%", "std", "max"}}
    summary = {
        "omics": {
            "rows": int(len(omics_df)),
            "features": int(len(omics_features)),
            "missing_values": int(omics_df.isna().sum().sum()),
        },
        "drugs": {
            "rows": int(len(drug_fp_df)),
            "features": int(len(drug_features)),
            "zero_fingerprint_frac": zero_fp,
            "missing_values": int(drug_fp_df.isna().sum().sum()),
        },
        "labels": {
            "rows": int(len(labels_df)),
            "unique_cell_lines": int(labels_df["cell_line"].nunique()),
            "unique_drugs": int(labels_df["drug"].nunique()),
            "ln_ic50_stats": label_stats,
            "missing_values": int(labels_df.isna().sum().sum()),
        },
        "metadata": {
            "rows": int(len(metadata_df)),
            "tissues": int(metadata_df["tissue"].nunique()) if "tissue" in metadata_df.columns else None,
            "missing_values": int(metadata_df.isna().sum().sum()),
        },
        "omics_features": omics_features,
        "drug_features": drug_features,
    }
    return summary


def _summarize_splits(splits_dir: Path) -> dict:
    summary = {}
    for split_path in sorted(splits_dir.glob("*_split.csv")):
        df = pd.read_csv(split_path)
        per_split = {}
        for split_name, sub in df.groupby("split"):
            per_split[str(split_name)] = {
                "rows": int(len(sub)),
                "cell_lines": int(sub["cell_line"].nunique()),
                "drugs": int(sub["drug"].nunique()),
            }
        summary[split_path.name] = {
            "rows": int(len(df)),
            "splits": per_split,
        }
    return summary


def main() -> None:
    setup_logging()
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    omics_df = load_parquet(processed_dir / "omics.parquet")
    drug_fp_df = load_parquet(processed_dir / "drug_fingerprints.parquet")
    labels_df = load_parquet(processed_dir / "labels.parquet")
    metadata_df = load_parquet(processed_dir / "metadata.parquet")

    manifest = _summarize_processed(omics_df, drug_fp_df, labels_df, metadata_df)
    splits_dir = Path(args.splits_dir) if args.splits_dir else processed_dir / "splits"
    if splits_dir.exists():
        manifest["splits_dir"] = str(splits_dir)
        manifest["splits"] = _summarize_splits(splits_dir)

    save_json(manifest, Path(args.out))


if __name__ == "__main__":
    main()
