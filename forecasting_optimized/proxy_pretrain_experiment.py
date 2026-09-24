from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from blend_models import as_distribution, tune_offsets
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, _amount_clusters, build_feature_tables


def add_candidate_families(frame: pd.DataFrame, repository: Path) -> pd.DataFrame:
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from dataset_cleaning.clean_dataset import classify_transaction

    result = frame.copy()
    result["candidate_family"] = [
        classify_transaction(mcc, description, amount)
        for mcc, description, amount in zip(
            result["mcc"], result["description"], result["amount"]
        )
    ]
    return result


def is_continuation(past: pd.DataFrame, future_row: object) -> bool:
    for stream in _amount_clusters(past):
        if len(stream) < 3:
            continue
        ordered = stream.sort_values("timestamp")
        gaps = ordered["timestamp"].diff().dt.total_seconds().dropna().to_numpy() / 86_400
        median_gap = float(np.median(gaps))
        if not 5 <= median_gap <= 120:
            continue
        gap_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) else np.inf
        if gap_cv > 0.75:
            continue
        amount_median = float(ordered["amount"].median())
        amount_difference = abs(float(future_row.amount) - amount_median) / max(
            abs(amount_median), 1.0
        )
        if amount_difference > 0.15:
            continue
        days_after_last = (
            future_row.timestamp - ordered["timestamp"].iloc[-1]
        ).total_seconds() / 86_400
        cycles = max(1, round(days_after_last / median_gap))
        phase_error = abs(days_after_last - cycles * median_gap)
        if phase_error <= max(8.0, 0.4 * median_gap):
            return True
    return False


def infer_proxy_targets(
    frame: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int = 90
) -> pd.Series:
    past = frame[frame["timestamp"] < cutoff]
    future = frame[
        (frame["timestamp"] >= cutoff)
        & (frame["timestamp"] < cutoff + pd.Timedelta(days=horizon_days))
    ]
    future_by_client = {
        client_id: group.sort_values("timestamp")
        for client_id, group in future.groupby("client_id", sort=False)
    }
    targets: dict[str, str] = {}
    for client_id, history in past.groupby("client_id", sort=False):
        target = "none"
        upcoming = future_by_client.get(client_id)
        if upcoming is not None:
            family_history = {
                family: group
                for family, group in history[
                    history["candidate_family"].isin(FAMILIES)
                ].groupby("candidate_family")
            }
            for row in upcoming.itertuples(index=False):
                family = str(row.candidate_family)
                if family in family_history and is_continuation(family_history[family], row):
                    target = family
                    break
        targets[str(client_id)] = target
    return pd.Series(targets, name="target")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Pretrain recurrence models on historical proxy labels")
    parser.add_argument(
        "--pretrain-jsonl",
        type=Path,
        default=repository / "data" / "dataset" / "unlabeled_pretrain_transactions.jsonl",
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=repository / "data" / "dataset_features"
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=("2025-08-01", "2025-09-01", "2025-10-01"),
    )
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    print("Loading optional pretraining transactions...")
    pretrain = pd.read_json(args.pretrain_jsonl, lines=True)
    pretrain["client_id"] = pretrain["client_id"].astype(str)
    pretrain["timestamp"] = pd.to_datetime(pretrain["timestamp"], utc=True)
    pretrain = add_candidate_families(pretrain, repository)
    proxy_wide_parts: list[pd.DataFrame] = []
    proxy_pair_parts: list[pd.DataFrame] = []
    proxy_target_parts: list[pd.Series] = []
    for cutoff_text in args.cutoffs:
        cutoff = pd.Timestamp(cutoff_text, tz="UTC")
        cutoff_targets = infer_proxy_targets(pretrain, cutoff)
        print(f"Proxy label distribution at {cutoff.date()}:")
        print(cutoff_targets.value_counts().reindex(LABELS, fill_value=0).to_string())
        if args.labels_only:
            continue
        suffix = f"@{cutoff.date()}"
        proxy_history = pretrain[pretrain["timestamp"] < cutoff].copy()
        proxy_history["client_id"] = proxy_history["client_id"] + suffix
        cutoff_targets.index = cutoff_targets.index + suffix
        cutoff_wide, cutoff_pairs = build_feature_tables(
            proxy_history, cutoff_targets.index, cutoff=cutoff
        )
        proxy_wide_parts.append(cutoff_wide)
        proxy_pair_parts.append(cutoff_pairs)
        proxy_target_parts.append(cutoff_targets)
    if args.labels_only:
        return
    proxy_wide = pd.concat(proxy_wide_parts)
    proxy_pairs = pd.concat(proxy_pair_parts)
    proxy_targets = pd.concat(proxy_target_parts)
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
    columns_wide = selected["wide_columns"]
    columns_pair = selected["pair_columns"]
    proxy_wide = proxy_wide.reindex(columns=columns_wide)
    train_wide = train_wide.reindex(columns=columns_wide)
    valid_wide = valid_wide.reindex(columns=columns_wide)
    proxy_pairs = proxy_pairs.reindex(columns=columns_pair)
    train_pairs = train_pairs.reindex(columns=columns_pair)
    valid_pairs = valid_pairs.reindex(columns=columns_pair)

    combined_wide = pd.concat([proxy_wide, train_wide])
    combined_targets = pd.concat([proxy_targets, train_targets])
    catboost_model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=700,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=6.0,
        random_strength=0.6,
        auto_class_weights="Balanced",
        random_seed=2026,
        allow_writing_files=False,
        verbose=False,
    )
    sample_weights = np.concatenate(
        [np.ones(len(proxy_wide)), np.full(len(train_wide), 4.0)]
    )
    catboost_model.fit(combined_wide, combined_targets.loc[combined_wide.index], sample_weight=sample_weights)
    catboost_probabilities = probabilities_in_label_order(catboost_model, valid_wide)
    catboost_score = f1_score(
        valid_targets.loc[valid_wide.index],
        np.asarray(LABELS)[catboost_probabilities.argmax(axis=1)],
        labels=LABELS,
        average="macro",
    )
    print(f"proxy-pretrained CatBoost macro_f1={catboost_score:.4f}")

    combined_pairs = pd.concat([proxy_pairs, train_pairs])
    combined_pair_targets = np.asarray(
        [combined_targets[client_id] == family for client_id, family in combined_pairs.index],
        dtype=np.int8,
    )
    pair_model = clone(selected["pair_model"])
    balanced = compute_sample_weight("balanced", combined_pair_targets)
    real_clients = set(train_targets.index)
    real_weight = np.asarray(
        [4.0 if client_id in real_clients else 1.0 for client_id, _ in combined_pairs.index]
    )
    pair_model.fit(
        combined_pairs,
        combined_pair_targets,
        histgradientboostingclassifier__sample_weight=balanced * real_weight,
    )
    pair_probabilities_frame = pair_probabilities(pair_model, valid_pairs).reindex(valid_targets.index)
    pair_distribution = as_distribution(pair_probabilities_frame.to_numpy())
    offsets, pair_score = tune_offsets(
        valid_targets.map({label: index for index, label in enumerate(LABELS)}).to_numpy(),
        pair_distribution,
    )
    predicted = (np.log(np.clip(pair_distribution, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(f"proxy-pretrained pair model calibrated macro_f1={pair_score:.4f}")
    print(
        classification_report(
            valid_targets,
            np.asarray(LABELS)[predicted],
            labels=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    joblib.dump(
        {
            "catboost_model": catboost_model,
            "pair_model": pair_model,
            "wide_columns": columns_wide,
            "pair_columns": columns_pair,
            "pair_offsets": offsets,
        },
        args.artifact_dir / "proxy_pretrained_models.joblib",
    )
    np.savez_compressed(
        args.artifact_dir / "proxy_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        catboost=catboost_probabilities,
        pair=pair_distribution,
    )


if __name__ == "__main__":
    main()
