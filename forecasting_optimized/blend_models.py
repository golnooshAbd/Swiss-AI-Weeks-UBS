from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables


def as_distribution(recurring: np.ndarray) -> np.ndarray:
    none = np.clip(1.0 - recurring.max(axis=1, keepdims=True), 1e-6, 1.0)
    probabilities = np.column_stack([recurring, none])
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def macro_f1(actual: np.ndarray, scores: np.ndarray, offsets: np.ndarray | None = None) -> float:
    adjusted = np.log(np.clip(scores, 1e-7, 1.0))
    if offsets is not None:
        adjusted = adjusted + offsets
    return float(
        f1_score(
            actual,
            adjusted.argmax(axis=1),
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )


def tune_offsets(actual: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    best_score = macro_f1(actual, probabilities, offsets)
    for step in (0.4, 0.2, 0.1, 0.05, 0.025):
        improved = True
        while improved:
            improved = False
            for column in range(len(LABELS)):
                for direction in (-step, step):
                    candidate = offsets.copy()
                    candidate[column] += direction
                    score = macro_f1(actual, probabilities, candidate)
                    if score > best_score + 1e-9:
                        offsets, best_score = candidate, score
                        improved = True
    return offsets, best_score


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Blend temporal and recurrence models")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=repository / "data" / "dataset_features" / "valid_features.csv",
    )
    parser.add_argument(
        "--labels", type=Path, default=repository / "data" / "dataset" / "valid_labels.csv"
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, dtype={"client_id": str})
    target_map = {name: index for index, name in enumerate(LABELS)}
    target = labels["target_next_recurring_merchant"].map(target_map)
    target_by_client = pd.Series(target.to_numpy(), index=labels["client_id"])

    bundle = joblib.load(args.artifact_dir / "model.joblib")
    frame = pd.read_csv(args.feature_csv)
    wide, pairs = build_feature_tables(frame, labels["client_id"])
    wide = wide.reindex(columns=bundle["wide_columns"])
    pairs = pairs.reindex(columns=bundle["pair_columns"])
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

    with np.load(args.artifact_dir / "family_valid_probabilities.npz") as family_data:
        family_by_client = {
            str(client_id): probabilities
            for client_id, probabilities in zip(
                family_data["client_ids"], family_data["probabilities"]
            )
        }
    family = as_distribution(np.stack([family_by_client[client_id] for client_id in wide.index]))

    with np.load(args.artifact_dir / "neural_valid_probabilities.npz") as neural_data:
        neural_by_client = {
            str(client_id): probabilities
            for client_id, probabilities in zip(
                neural_data["client_ids"], neural_data["probabilities"]
            )
        }
    neural = np.stack([neural_by_client[client_id] for client_id in wide.index])
    actual = target_by_client.loc[wide.index].to_numpy()

    best = (-1.0, (0.0, 0.0, 0.0, 0.0), np.zeros(len(LABELS)), catboost)
    for catboost_weight in np.linspace(0.0, 1.0, 11):
        for pair_weight in np.linspace(0.0, 1.0 - catboost_weight, 11):
            remaining = 1.0 - catboost_weight - pair_weight
            for family_share in np.linspace(0.0, 1.0, 6):
                family_weight = remaining * family_share
                neural_weight = remaining - family_weight
                probabilities = (
                    catboost_weight * catboost
                    + pair_weight * pair
                    + family_weight * family
                    + neural_weight * neural
                )
                offsets, score = tune_offsets(actual, probabilities)
                if score > best[0]:
                    best = (
                        score,
                        (
                            float(catboost_weight),
                            float(pair_weight),
                            float(family_weight),
                            float(neural_weight),
                        ),
                        offsets,
                        probabilities,
                    )

    score, weights, offsets, probabilities = best
    extra_weights: dict[str, float] = {}
    extra_sources: dict[str, np.ndarray] = {}
    stacking_path = args.artifact_dir / "stacking_valid_probabilities.npz"
    if stacking_path.exists():
        with np.load(stacking_path) as data:
            by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["stacking"] = np.stack([by_client[client_id] for client_id in wide.index])
    proxy_path = args.artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(value) for value in data["client_ids"]]
            positions = {client_id: index for index, client_id in enumerate(proxy_order)}
            extra_sources["proxy_catboost"] = np.stack(
                [data["catboost"][positions[client_id]] for client_id in wide.index]
            )
            extra_sources["proxy_pair"] = np.stack(
                [data["pair"][positions[client_id]] for client_id in wide.index]
            )
    xgboost_path = args.artifact_dir / "xgboost_valid_probabilities.npz"
    if xgboost_path.exists():
        with np.load(xgboost_path) as data:
            xgboost_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["xgboost_pair"] = np.stack(
            [xgboost_by_client[client_id] for client_id in wide.index]
        )
    ranking_path = args.artifact_dir / "ranking_valid_probabilities.npz"
    if ranking_path.exists():
        with np.load(ranking_path) as data:
            ranking_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["ranking"] = np.stack(
            [ranking_by_client[client_id] for client_id in wide.index]
        )
    embedding_path = args.artifact_dir / "embedding_pool_valid_probabilities.npz"
    if embedding_path.exists():
        with np.load(embedding_path) as data:
            embedding_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["embedding_pool"] = np.stack(
            [embedding_by_client[client_id] for client_id in wide.index]
        )
    description_path = args.artifact_dir / "description_stream_valid_probabilities.npz"
    if description_path.exists():
        with np.load(description_path) as data:
            description_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["description_stream"] = np.stack(
            [description_by_client[client_id] for client_id in wide.index]
        )
    catboost_tuned_path = args.artifact_dir / "catboost_tuned_valid_probabilities.npz"
    if catboost_tuned_path.exists():
        with np.load(catboost_tuned_path) as data:
            ct_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["catboost_tuned"] = np.stack(
            [ct_by_client[client_id] for client_id in wide.index]
        )
    pair_tuned_path = args.artifact_dir / "pair_tuned_valid_probabilities.npz"
    if pair_tuned_path.exists():
        with np.load(pair_tuned_path) as data:
            pt_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["pair_tuned"] = np.stack(
            [pt_by_client[client_id] for client_id in wide.index]
        )
    tfidf_path = args.artifact_dir / "tfidf_valid_probabilities.npz"
    if tfidf_path.exists():
        with np.load(tfidf_path) as data:
            tf_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"]
                )
            }
        extra_sources["tfidf"] = np.stack(
            [tf_by_client[client_id] for client_id in wide.index]
        )

    for source_name, source_probabilities in extra_sources.items():
        source_best = (score, 0.0, offsets, probabilities)
        for source_weight in np.linspace(0.02, 0.5, 25):
            candidate_probabilities = np.exp(
                (1 - source_weight) * np.log(probabilities + 1e-7)
                + source_weight * np.log(source_probabilities + 1e-7)
            )
            candidate_probabilities /= candidate_probabilities.sum(axis=1, keepdims=True)
            candidate_offsets, candidate_score = tune_offsets(
                actual, candidate_probabilities
            )
            if candidate_score > source_best[0]:
                source_best = (
                    candidate_score,
                    float(source_weight),
                    candidate_offsets,
                    candidate_probabilities,
                )
        if source_best[1] > 0:
            score, extra_weights[source_name], offsets, probabilities = source_best

    # A small second-stage calibration lets a specialist affect only the class it
    # ranks well. The values are deliberately restricted and applied only when the
    # held-out macro-F1 increases. Negative values de-correlate systematic errors.
    proposed_adjustments = {
        "stacking": {"gym": 0.10, "music": 0.125, "software": 0.40, "none": 0.225},
        "xgboost_pair": {
            "cloud": -0.15,
            "gym": 0.35,
            "insurance": -0.125,
            "mobile": -0.025,
            "software": -0.20,
            "streaming": -0.025,
        },
        "ranking": {"mobile": 0.025, "software": 0.025, "streaming": 0.05},
        "embedding_pool": {"cloud": 0.075, "mobile": 0.175, "streaming": 0.475},
        "description_stream": {"cloud": -0.025, "streaming": 0.05},
    }
    class_adjustments: list[dict[str, float | str]] = []
    for source_name, adjustments in proposed_adjustments.items():
        if source_name not in extra_sources:
            continue
        source_probabilities = extra_sources[source_name]
        for label, weight in adjustments.items():
            column = LABELS.index(label)
            candidate_probabilities = probabilities.copy()
            candidate_probabilities[:, column] = (
                (1 - weight) * probabilities[:, column]
                + weight * source_probabilities[:, column]
            )
            candidate_probabilities = np.clip(candidate_probabilities, 1e-7, None)
            candidate_probabilities /= candidate_probabilities.sum(axis=1, keepdims=True)
            candidate_offsets, candidate_score = tune_offsets(actual, candidate_probabilities)
            if candidate_score > score + 1e-9:
                score = candidate_score
                probabilities = candidate_probabilities
                offsets = candidate_offsets
                class_adjustments.append(
                    {"source": source_name, "label": label, "weight": weight}
                )

    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    print(f"best blended validation macro_f1={score:.4f}")
    print(f"weights catboost/pair/family/neural={weights}")
    print(f"extra sequential blend weights={extra_weights}")
    print(f"class-specific adjustments={class_adjustments}")
    print(f"class offsets={dict(zip(LABELS, offsets.round(3)))}")
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
    configuration = {
        "macro_f1": score,
        "weights": dict(
            zip(("catboost", "pair", "family", "neural"), weights)
        ),
        "class_offsets": dict(zip(LABELS, offsets.tolist())),
        "extra_weights": extra_weights,
        "class_adjustments": class_adjustments,
    }
    (args.artifact_dir / "blend_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "client_id": wide.index,
            "actual": np.asarray(LABELS)[actual],
            "predicted": np.asarray(LABELS)[predicted],
        }
    ).to_csv(args.artifact_dir / "blended_validation_predictions.csv", index=False)
    np.savez_compressed(
        args.artifact_dir / "blended_valid_probabilities.npz",
        client_ids=np.asarray(wide.index, dtype=str),
        probabilities=probabilities,
        offsets=offsets,
    )


if __name__ == "__main__":
    main()
