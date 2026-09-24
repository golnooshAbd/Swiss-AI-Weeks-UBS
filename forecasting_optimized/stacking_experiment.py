from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from blend_models import as_distribution, tune_offsets
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables


def meta_features(pair_recurring: np.ndarray, multiclass: np.ndarray) -> np.ndarray:
    pair_distribution = as_distribution(pair_recurring)
    sorted_pair = np.sort(pair_recurring, axis=1)
    summary = np.column_stack(
        [
            pair_recurring.max(axis=1),
            sorted_pair[:, -2],
            sorted_pair[:, -1] - sorted_pair[:, -2],
            pair_recurring.sum(axis=1),
            multiclass.max(axis=1),
        ]
    )
    return np.column_stack(
        [
            pair_recurring,
            pair_distribution,
            multiclass,
            pair_distribution - multiclass,
            summary,
        ]
    )


def report(name: str, actual: np.ndarray, probabilities: np.ndarray) -> float:
    predicted = probabilities.argmax(axis=1)
    score = float(
        f1_score(
            actual,
            predicted,
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )
    print(f"{name}: macro_f1={score:.4f}")
    return score


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Leakage-free out-of-fold stacking experiment")
    parser.add_argument(
        "--feature-dir", type=Path, default=repository / "data" / "dataset_features"
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train_labels = pd.read_csv(args.label_dir / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(args.label_dir / "valid_labels.csv", dtype={"client_id": str})
    target_index = {name: index for index, name in enumerate(LABELS)}
    train_target_names = train_labels["target_next_recurring_merchant"].to_numpy()
    train_target_by_client = train_labels.set_index("client_id")[
        "target_next_recurring_merchant"
    ]
    train_target = train_labels["target_next_recurring_merchant"].map(target_index).to_numpy()
    valid_target = valid_labels["target_next_recurring_merchant"].map(target_index).to_numpy()

    train_wide, train_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "train_features.csv"), train_labels["client_id"]
    )
    valid_wide, valid_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "valid_features.csv"), valid_labels["client_id"]
    )
    selected = joblib.load(args.artifact_dir / "model.joblib")
    train_wide = train_wide.reindex(columns=selected["wide_columns"])
    valid_wide = valid_wide.reindex(columns=selected["wide_columns"])
    train_pairs = train_pairs.reindex(columns=selected["pair_columns"])
    valid_pairs = valid_pairs.reindex(columns=selected["pair_columns"])

    oof_pair = np.zeros((len(train_labels), len(FAMILIES)), dtype=np.float64)
    oof_catboost = np.zeros((len(train_labels), len(LABELS)), dtype=np.float64)
    valid_pair_folds: list[np.ndarray] = []
    valid_catboost_folds: list[np.ndarray] = []
    pair_models: list[object] = []
    catboost_models: list[CatBoostClassifier] = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=2026)

    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(train_wide, train_target), start=1
    ):
        fit_clients = train_labels.iloc[fit_indices]["client_id"]
        holdout_clients = train_labels.iloc[holdout_indices]["client_id"]
        fit_client_set = set(fit_clients)
        holdout_client_set = set(holdout_clients)
        pair_fit_mask = train_pairs.index.get_level_values("client_id").isin(fit_client_set)
        pair_holdout_mask = train_pairs.index.get_level_values("client_id").isin(
            holdout_client_set
        )
        pair_fit = train_pairs[pair_fit_mask]
        pair_holdout = train_pairs[pair_holdout_mask]
        pair_fit_target = np.asarray(
            [train_target_by_client[client_id] == family for client_id, family in pair_fit.index],
            dtype=np.int8,
        )
        pair_model = clone(selected["pair_model"])
        weights = compute_sample_weight("balanced", pair_fit_target)
        pair_model.fit(
            pair_fit,
            pair_fit_target,
            histgradientboostingclassifier__sample_weight=weights,
        )
        holdout_scores = pair_probabilities(pair_model, pair_holdout).reindex(holdout_clients)
        oof_pair[holdout_indices] = holdout_scores.to_numpy()
        valid_pair_folds.append(
            pair_probabilities(pair_model, valid_pairs)
            .reindex(valid_labels["client_id"])
            .to_numpy()
        )
        pair_models.append(pair_model)

        catboost_model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=476,
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            auto_class_weights="Balanced",
            random_seed=2026 + fold,
            allow_writing_files=False,
            verbose=False,
        )
        catboost_model.fit(
            train_wide.iloc[fit_indices],
            train_target_names[fit_indices],
            verbose=False,
        )
        oof_catboost[holdout_indices] = probabilities_in_label_order(
            catboost_model, train_wide.iloc[holdout_indices]
        )
        valid_catboost_folds.append(
            probabilities_in_label_order(catboost_model, valid_wide)
        )
        catboost_models.append(catboost_model)
        print(f"completed fold {fold}/{args.folds}")

    valid_pair = np.mean(valid_pair_folds, axis=0)
    valid_catboost = np.mean(valid_catboost_folds, axis=0)
    train_meta = meta_features(oof_pair, oof_catboost)
    valid_meta = meta_features(valid_pair, valid_catboost)
    candidates: list[tuple[str, object, np.ndarray, float]] = []

    report("fold-averaged pair rule", valid_target, as_distribution(valid_pair))
    report("fold-averaged CatBoost", valid_target, valid_catboost)

    for regularization in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=2026,
        )
        model.fit(train_meta, train_target)
        probabilities = model.predict_proba(valid_meta)
        score = report(f"stacked logistic C={regularization}", valid_target, probabilities)
        candidates.append((f"logistic_{regularization}", model, probabilities, score))

    for leaves, minimum_leaf in ((7, 30), (15, 25), (23, 20), (31, 20)):
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=leaves,
            min_samples_leaf=minimum_leaf,
            l2_regularization=2.0,
            random_state=2026,
        )
        weights = compute_sample_weight("balanced", train_target)
        model.fit(train_meta, train_target, sample_weight=weights)
        probabilities = model.predict_proba(valid_meta)
        score = report(
            f"stacked gradient leaves={leaves} min_leaf={minimum_leaf}",
            valid_target,
            probabilities,
        )
        candidates.append((f"gradient_{leaves}_{minimum_leaf}", model, probabilities, score))

    name, meta_model, probabilities, uncalibrated_score = max(
        candidates, key=lambda value: value[3]
    )
    offsets, calibrated_score = tune_offsets(valid_target, probabilities)
    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(
        f"\nbest stack={name} uncalibrated={uncalibrated_score:.4f} "
        f"calibrated={calibrated_score:.4f}"
    )
    print(classification_report(valid_target, predicted, target_names=LABELS, digits=3))
    joblib.dump(
        {
            "pair_models": pair_models,
            "catboost_models": catboost_models,
            "meta_model": meta_model,
            "meta_name": name,
            "offsets": offsets,
            "wide_columns": selected["wide_columns"],
            "pair_columns": selected["pair_columns"],
        },
        args.artifact_dir / "stacking_model.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "stacking_valid_probabilities.npz",
        client_ids=np.asarray(valid_labels["client_id"], dtype=str),
        probabilities=probabilities,
        calibrated_predictions=predicted,
    )


if __name__ == "__main__":
    main()
