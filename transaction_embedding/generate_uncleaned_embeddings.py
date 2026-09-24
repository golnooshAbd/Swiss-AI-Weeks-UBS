from __future__ import annotations

import argparse
from pathlib import Path

from txembed.generation import generate_embedding_artifacts


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    features = repository / "data" / "dataset_features"
    parser = argparse.ArgumentParser(
        description="Embed original descriptions while retaining every transaction and feature"
    )
    parser.add_argument("--train-csv", type=Path, default=features / "train_features.csv")
    parser.add_argument("--valid-csv", type=Path, default=features / "valid_features.csv")
    parser.add_argument("--test-csv", type=Path, default=features / "test_features.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "artifacts" / "uncleaned"
    )
    parser.add_argument("--cache", type=Path, help="Defaults to OUTPUT_DIR/descriptions.sqlite")
    parser.add_argument("--device", help="Sentence Transformer device, e.g. cpu, cuda, or mps")
    args = parser.parse_args()

    generate_embedding_artifacts(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        description_column="description",
        cache_path=args.cache,
        device=args.device,
    )


if __name__ == "__main__":
    main()
