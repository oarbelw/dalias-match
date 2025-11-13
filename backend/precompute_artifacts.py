"""Utility script to pre-compute recommender artifacts offline.

Run locally (not inside Cloud Run/Railway) to generate an ``artifacts.pkl`` file
that can be uploaded to GitHub Releases, Cloud Storage, etc.

Usage::

    python backend/precompute_artifacts.py \
        --dataset path/to/ratings_df.parquet \
        --output artifacts.pkl

The resulting pickle can be hosted publicly and referenced via the
``ARTIFACTS_URL`` environment variable (or the default release URL).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from backend.core_recommender import build_model_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute recommender artifacts for deployment.")
    parser.add_argument(
        "--dataset",
        default="ratings_df.parquet",
        help="Path to the ratings parquet file (default: ratings_df.parquet)",
    )
    parser.add_argument(
        "--output",
        default="artifacts.pkl",
        help="Path to write the pickle file (default: artifacts.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    df["rating_val"] = df["rating_val"].astype("float32")
    df["year_released"] = df["year_released"].astype("float32")

    artifacts = build_model_artifacts(df)

    output_path = Path(args.output)
    output_path.write_bytes(pickle.dumps(artifacts))
    print(f"Saved artifacts to {output_path.resolve()}")


if __name__ == "__main__":
    main()
