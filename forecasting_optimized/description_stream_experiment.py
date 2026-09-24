from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from blend_models import as_distribution, tune_offsets
from experiment import pair_probabilities
from features import (
    CUTOFF,
    FAMILIES,
    LABELS,
    STREAM_FEATURES,
    _transaction_stats_without_streams,
    build_feature_tables,
)


DECORATION = re.compile(
    r"\b(?:billing|member|pay|online|service|core|digital|dgtl|plus)\b"
)


def canonical_description(value: object) -> str:
    """Collapse synthetic description decorations while preserving the merchant phrase."""
    normalized = DECORATION.sub(" ", str(value).lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "generic"


def description_features(
    frame: pd.DataFrame, client_ids: pd.Index, cutoff: pd.Timestamp = CUTOFF
) -> pd.DataFrame:
    data = frame.copy()
    data["client_id"] = data["client_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["canonical_description"] = data["clean_description"].map(canonical_description)
    rows: list[dict[str, float | str]] = []
    by_client = {key: value for key, value in data.groupby("client_id", sort=False)}
    for client_id in client_ids.astype(str):
        client = by_client[client_id]
        for family in FAMILIES:
            family_rows = client[client["candidate_family"] == family]
            streams: list[dict[str, float]] = []
            for _, stream in family_rows.groupby("canonical_description", sort=False):
                if len(stream) >= 2:
                    streams.append(_transaction_stats_without_streams(stream, cutoff))
            streams.sort(
                key=lambda values: (
                    -values["count"],
                    values["days_since_last"],
                    values.get("gap_cv", np.inf),
                )
            )
            row: dict[str, float | str] = {
                "client_id": client_id,
                "family": family,
                "description_stream_count": float(len(streams)),
            }
            for rank, stats in enumerate(streams[:4]):
                for name in STREAM_FEATURES:
                    row[f"description_stream_{rank}__{name}"] = stats.get(name, np.nan)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["client_id", "family"])


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Description-normalized recurrence model")
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
    train = pd.read_csv(args.feature_dir / "train_features.csv")
    valid = pd.read_csv(args.feature_dir / "valid_features.csv")
    _, train_pairs = build_feature_tables(train, train_targets.index)
    _, valid_pairs = build_feature_tables(valid, valid_targets.index)
    train_description = description_features(train, train_targets.index)
    valid_description = description_features(valid, valid_targets.index)
    train_pairs = train_pairs.join(train_description)
    valid_pairs = valid_pairs.join(valid_description)
    columns = sorted(set(train_pairs) | set(valid_pairs))
    train_pairs = train_pairs.reindex(columns=columns)
    valid_pairs = valid_pairs.reindex(columns=columns)

    train_binary = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_pairs.index],
        dtype=np.int8,
    )
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=10.0,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=4.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(
        train_pairs,
        train_binary,
        sample_weight=compute_sample_weight("balanced", train_binary),
        verbose=False,
    )
    recurring = pair_probabilities(model, valid_pairs).reindex(valid_targets.index).to_numpy()
    probabilities = as_distribution(recurring)
    actual = valid_targets.map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    offsets, score = tune_offsets(actual, probabilities)
    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(f"description-stream model macro_f1={score:.4f}")
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
        {"model": model, "columns": columns, "offsets": offsets},
        args.artifact_dir / "description_stream_model.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "description_stream_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()
