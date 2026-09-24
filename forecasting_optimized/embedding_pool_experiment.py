from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from blend_models import tune_offsets
from features import LABELS


def aggregate_clients(
    artifact_dir: Path, split: str, preprocessor: Any
) -> tuple[np.ndarray, np.ndarray]:
    # Importing the PyTorch-backed embedding package and XGBoost into the same
    # process can conflict in native OpenMP runtimes on macOS. Preparation is
    # therefore run in a short child process; model fitting reads the cache.
    from txembed import PreprocessedTransactions, TransactionSequenceDataset

    rows = PreprocessedTransactions.load_npz(artifact_dir / f"{split}.npz")
    sequences = TransactionSequenceDataset(rows)
    cardinalities = [
        preprocessor.categorical_cardinalities[name]
        for name in preprocessor.schema.categorical
    ]
    output: list[np.ndarray] = []
    for indices in sequences.indices:
        description = rows.description_embeddings[indices]
        dense = rows.dense_features[indices]
        categorical = rows.categorical_indices[indices]
        recent_description = description[-min(12, len(description)) :]
        recent_dense = dense[-min(12, len(dense)) :]
        parts = [
            description.mean(axis=0),
            description.std(axis=0),
            description[-1],
            recent_description.mean(axis=0),
            dense.mean(axis=0),
            dense.std(axis=0),
            dense[-1],
            recent_dense.mean(axis=0),
            np.asarray([len(indices)], dtype=np.float32),
        ]
        for column, cardinality in enumerate(cardinalities):
            counts = np.bincount(
                categorical[:, column], minlength=cardinality
            ).astype(np.float32)
            parts.append(counts / len(indices))
        output.append(np.concatenate(parts))
    return np.asarray(sequences.client_ids, dtype=str), np.stack(output).astype(np.float32)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Boosted model over pooled transaction embeddings")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=repository / "transaction_embedding" / "artifacts" / "uncleaned",
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    cache_path = args.artifact_dir / "embedding_pool_features.npz"
    if args.prepare_only:
        from txembed import TransactionPreprocessor

        preprocessor = TransactionPreprocessor.load(args.embedding_dir / "preprocessor.json")
        train_clients, train_features = aggregate_clients(args.embedding_dir, "train", preprocessor)
        valid_clients, valid_features = aggregate_clients(args.embedding_dir, "valid", preprocessor)
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_clients=train_clients,
            train_features=train_features,
            valid_clients=valid_clients,
            valid_features=valid_features,
        )
        return
    if not cache_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--embedding-dir",
                str(args.embedding_dir),
                "--artifact-dir",
                str(args.artifact_dir),
                "--prepare-only",
            ],
            check=True,
        )
    cached = np.load(cache_path)
    train_clients = cached["train_clients"]
    train_features = cached["train_features"]
    valid_clients = cached["valid_clients"]
    valid_features = cached["valid_features"]
    label_index = {name: index for index, name in enumerate(LABELS)}
    train_labels = pd.read_csv(args.label_dir / "train_labels.csv", dtype={"client_id": str}).set_index(
        "client_id"
    )
    valid_labels = pd.read_csv(args.label_dir / "valid_labels.csv", dtype={"client_id": str}).set_index(
        "client_id"
    )
    train_target = train_labels.loc[train_clients, "target_next_recurring_merchant"].map(
        label_index
    ).to_numpy(dtype=np.int64)
    valid_target = valid_labels.loc[valid_clients, "target_next_recurring_merchant"].map(
        label_index
    ).to_numpy(dtype=np.int64)
    train_features = np.ascontiguousarray(train_features, dtype=np.float32)
    valid_features = np.ascontiguousarray(valid_features, dtype=np.float32)
    weights = np.asarray(compute_sample_weight("balanced", train_target), dtype=np.float32)

    best: tuple[float, XGBClassifier, np.ndarray, np.ndarray, tuple] | None = None
    for depth, child_weight, column_fraction in (
        (3, 5.0, 0.5),
        (4, 8.0, 0.6),
        (5, 12.0, 0.7),
    ):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            n_estimators=1600,
            learning_rate=0.025,
            max_depth=depth,
            min_child_weight=child_weight,
            subsample=0.85,
            colsample_bytree=column_fraction,
            reg_alpha=0.2,
            reg_lambda=4.0,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=120,
            random_state=2026,
            n_jobs=-1,
        )
        model.fit(
            train_features,
            train_target,
            sample_weight=weights,
            eval_set=[(valid_features, valid_target)],
            verbose=False,
        )
        probabilities = model.predict_proba(valid_features)
        offsets, score = tune_offsets(valid_target, probabilities)
        print(
            f"depth={depth} child_weight={child_weight} best_iteration={model.best_iteration} "
            f"calibrated_macro_f1={score:.4f}"
        )
        candidate = (score, model, probabilities, offsets, (depth, child_weight, column_fraction))
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    score, model, probabilities, offsets, configuration = best
    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(f"\nbest pooled-embedding model={configuration} macro_f1={score:.4f}")
    print(
        classification_report(
            valid_target,
            predicted,
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    joblib.dump(
        {"model": model, "offsets": offsets},
        args.artifact_dir / "embedding_pool_model.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "embedding_pool_valid_probabilities.npz",
        client_ids=valid_clients,
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()
