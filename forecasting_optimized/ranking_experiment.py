from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRanker

from blend_models import tune_offsets
from features import FAMILIES, LABELS, build_feature_tables


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=1, keepdims=True)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Separate recurrence and family-ranking models")
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
    train_wide, train_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "train_features.csv"), train_targets.index
    )
    valid_wide, valid_pairs = build_feature_tables(
        pd.read_csv(args.feature_dir / "valid_features.csv"), valid_targets.index
    )
    selected = joblib.load(args.artifact_dir / "model.joblib")
    train_wide = train_wide.reindex(columns=selected["wide_columns"])
    valid_wide = valid_wide.reindex(columns=selected["wide_columns"])
    train_pairs = train_pairs.reindex(columns=selected["pair_columns"])
    valid_pairs = valid_pairs.reindex(columns=selected["pair_columns"])

    recurring_clients = train_targets[train_targets != "none"].index
    ranking_train = train_pairs[
        train_pairs.index.get_level_values("client_id").isin(recurring_clients)
    ]
    ranking_target = np.asarray(
        [train_targets[client_id] == family for client_id, family in ranking_train.index],
        dtype=np.int8,
    )
    ranker = XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@1",
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        tree_method="hist",
        random_state=2026,
        n_jobs=-1,
    )
    ranker.fit(
        ranking_train,
        ranking_target,
        group=np.full(len(recurring_clients), len(FAMILIES), dtype=np.int32),
        verbose=False,
    )
    ranking_scores = ranker.predict(valid_pairs).reshape(len(valid_targets), len(FAMILIES))

    train_none = (train_targets.loc[train_wide.index] == "none").astype(np.int8)
    valid_none = (valid_targets.loc[valid_wide.index] == "none").astype(np.int8)
    none_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1600,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=10.0,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.15,
        reg_lambda=3.0,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=120,
        random_state=2026,
        n_jobs=-1,
    )
    none_model.fit(
        train_wide,
        train_none,
        sample_weight=compute_sample_weight("balanced", train_none),
        eval_set=[(valid_wide, valid_none)],
        verbose=False,
    )
    none_probability = none_model.predict_proba(valid_wide)[:, 1]
    actual = valid_targets.map({label: index for index, label in enumerate(LABELS)}).to_numpy()

    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for temperature in np.linspace(0.2, 2.0, 19):
        family_conditional = softmax(ranking_scores, float(temperature))
        probabilities = np.column_stack(
            [family_conditional * (1 - none_probability[:, None]), none_probability]
        )
        offsets, score = tune_offsets(actual, probabilities)
        candidate = (score, float(temperature), offsets, probabilities)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    score, temperature, offsets, probabilities = best
    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    recurring_mask = actual != len(LABELS) - 1
    family_accuracy = float(
        np.mean(probabilities[recurring_mask, : len(FAMILIES)].argmax(axis=1) == actual[recurring_mask])
    )
    print(
        f"ranking model macro_f1={score:.4f} conditional_family_accuracy={family_accuracy:.4f} "
        f"temperature={temperature:.2f}"
    )
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
            "ranker": ranker,
            "none_model": none_model,
            "temperature": temperature,
            "offsets": offsets,
            "wide_columns": selected["wide_columns"],
            "pair_columns": selected["pair_columns"],
        },
        args.artifact_dir / "ranking_model.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "ranking_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()
