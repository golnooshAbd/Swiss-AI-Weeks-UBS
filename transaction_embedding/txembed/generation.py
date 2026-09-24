from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import DescriptionEmbeddingCache
from .preprocessing import TransactionPreprocessor
from .schema import FeatureSchema


def generate_embedding_artifacts(
    *,
    train_csv: Path,
    valid_csv: Path,
    test_csv: Path,
    output_dir: Path,
    description_column: str,
    cache_path: Path | None = None,
    device: str | None = None,
    encoder_factory: Callable[[str], Any] | None = None,
) -> dict[str, int]:
    """Generate equivalent artifacts while choosing which description text to embed.

    All three inputs must already contain the complete engineered feature schema.
    The function changes only the text column used by MiniLM and verifies that every
    input transaction is represented in the resulting artifact.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = DescriptionEmbeddingCache(
        cache_path or output_dir / "descriptions.sqlite",
        device=device,
        encoder_factory=encoder_factory,
    )
    preprocessor = TransactionPreprocessor(FeatureSchema(description=description_column))
    paths = {"train": train_csv, "valid": valid_csv, "test": test_csv}
    counts: dict[str, int] = {}

    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} feature CSV: {path}")

        frame = pd.read_csv(path)
        if split == "train":
            rows = preprocessor.fit_transform(frame, cache)
            preprocessor.save(output_dir / "preprocessor.json")
        else:
            rows = preprocessor.transform(frame, cache)

        if len(rows) != len(frame):
            raise RuntimeError(
                f"{split} lost transactions: input has {len(frame):,} rows, "
                f"artifact has {len(rows):,}"
            )

        rows.save_npz(output_dir / f"{split}.npz")
        counts[split] = len(rows)
        print(f"{split}: {len(rows):,} transactions -> {output_dir / f'{split}.npz'}")

    return counts
