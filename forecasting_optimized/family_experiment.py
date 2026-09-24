from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score

from experiment import predict_pairs, tune_pair_threshold
from features import FAMILIES, LABELS, build_feature_tables


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train one recurrence model per merchant family")
    parser.add_argument(
        "--feature-dir", type=Path, default=repository / "data" / "dataset_features"
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_labels = pd.read_csv(args.label_dir / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(args.label_dir / "valid_labels.csv", dtype={"client_id": str})
    train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
    valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]
    _, train_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "train_features.csv"), train_labels["client_id"]
    )
    _, valid_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "valid_features.csv"), valid_labels["client_id"]
    )
    columns = sorted(set(train_pairs) | set(valid_pairs))
    train_pairs = train_pairs.reindex(columns=columns)
    valid_pairs = valid_pairs.reindex(columns=columns)

    models: dict[str, CatBoostClassifier] = {}
    probabilities = pd.DataFrame(index=valid_labels["client_id"], columns=FAMILIES, dtype=float)
    for family in FAMILIES:
        train_family = train_pairs.xs(family, level="family").reindex(train_labels["client_id"])
        valid_family = valid_pairs.xs(family, level="family").reindex(valid_labels["client_id"])
        train_binary = (train_targets.reindex(train_labels["client_id"]) == family).astype(int)
        valid_binary = (valid_targets.reindex(valid_labels["client_id"]) == family).astype(int)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1000,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=6.0,
            random_strength=0.7,
            auto_class_weights="Balanced",
            random_seed=2026,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            train_family,
            train_binary,
            eval_set=(valid_family, valid_binary),
            early_stopping_rounds=100,
            verbose=False,
        )
        models[family] = model
        probabilities[family] = model.predict_proba(valid_family)[:, 1]
        print(f"{family}: best_iteration={model.get_best_iteration()}")

    threshold, score = tune_pair_threshold(probabilities, valid_targets)
    predicted = predict_pairs(probabilities, threshold)
    actual = valid_targets.reindex(probabilities.index).to_numpy()
    print(f"\nfamily-specific models threshold={threshold:.3f} macro_f1={score:.4f}")
    print(classification_report(actual, predicted, labels=LABELS, digits=3, zero_division=0))
    joblib.dump(
        {"models": models, "columns": columns, "threshold": threshold},
        args.output_dir / "family_models.joblib",
    )
    np.savez_compressed(
        args.output_dir / "family_valid_probabilities.npz",
        client_ids=np.asarray(probabilities.index, dtype=str),
        probabilities=probabilities.to_numpy(),
    )


if __name__ == "__main__":
    main()
