from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from blend_models import as_distribution, tune_offsets
from experiment import pair_probabilities
from features import FAMILIES, LABELS, build_feature_tables


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="XGBoost recurrence experiments")
    parser.add_argument(
        "--feature-dir", type=Path, default=repository / "data" / "dataset_features"
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()

    train_labels = pd.read_csv(args.label_dir / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(args.label_dir / "valid_labels.csv", dtype={"client_id": str})
    train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
    valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]
    _, train_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "train_features.csv"), train_targets.index
    )
    _, valid_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "valid_features.csv"), valid_targets.index
    )
    selected = joblib.load(args.artifact_dir / "model.joblib")
    train_pairs = train_pairs.reindex(columns=selected["pair_columns"])
    valid_pairs = valid_pairs.reindex(columns=selected["pair_columns"])
    train_binary = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_pairs.index],
        dtype=np.int8,
    )
    valid_binary = np.asarray(
        [valid_targets[client_id] == family for client_id, family in valid_pairs.index],
        dtype=np.int8,
    )
    sample_weights = compute_sample_weight("balanced", train_binary)
    actual = valid_targets.map({label: index for index, label in enumerate(LABELS)}).to_numpy()

    configurations = (
        (3, 5.0, 0.8),
        (4, 8.0, 0.85),
        (5, 12.0, 0.85),
        (6, 18.0, 0.9),
    )
    best: tuple[float, XGBClassifier, np.ndarray, np.ndarray, tuple] | None = None
    for depth, minimum_child_weight, subsample in configurations:
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1800,
            learning_rate=0.025,
            max_depth=depth,
            min_child_weight=minimum_child_weight,
            subsample=subsample,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=3.0,
            gamma=0.02,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=120,
            random_state=2026,
            n_jobs=-1,
        )
        model.fit(
            train_pairs,
            train_binary,
            sample_weight=sample_weights,
            eval_set=[(valid_pairs, valid_binary)],
            verbose=False,
        )
        recurring = pair_probabilities(model, valid_pairs).reindex(valid_targets.index).to_numpy()
        distribution = as_distribution(recurring)
        offsets, score = tune_offsets(actual, distribution)
        print(
            f"depth={depth} min_child={minimum_child_weight} "
            f"best_iteration={model.best_iteration} calibrated_macro_f1={score:.4f}"
        )
        candidate = (score, model, distribution, offsets, (depth, minimum_child_weight, subsample))
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    score, model, distribution, offsets, configuration = best
    predicted = (np.log(np.clip(distribution, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(f"\nbest XGBoost pair configuration={configuration} macro_f1={score:.4f}")
    print(
        classification_report(
            actual,
            predicted,
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    joblib.dump(
        {
            "model": model,
            "columns": selected["pair_columns"],
            "offsets": offsets,
        },
        args.artifact_dir / "xgboost_pair_model.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "xgboost_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        probabilities=distribution,
    )


if __name__ == "__main__":
    main()
