from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from features import FAMILIES, LABELS, build_feature_tables, build_stream_table


def load_targets(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"client_id": str})
    return frame.set_index("client_id")["target_next_recurring_merchant"].astype(str)


def probabilities_in_label_order(model: object, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    return np.column_stack([probabilities[:, classes.index(label)] for label in LABELS])


def pair_probabilities(model: object, features: pd.DataFrame) -> pd.DataFrame:
    positive_index = list(model.classes_).index(1)
    values = model.predict_proba(features)[:, positive_index]
    client_order = pd.Index(features.index.get_level_values("client_id")).drop_duplicates()
    return (
        pd.Series(values, index=features.index)
        .unstack("family")
        .reindex(index=client_order, columns=FAMILIES)
    )


def predict_pairs(probabilities: pd.DataFrame, threshold: float) -> np.ndarray:
    best_family = probabilities.idxmax(axis=1).to_numpy()
    best_score = probabilities.max(axis=1).to_numpy()
    return np.where(best_score >= threshold, best_family, "none")


def stream_probabilities(model: object, features: pd.DataFrame) -> pd.DataFrame:
    positive_index = list(model.classes_).index(1)
    values = model.predict_proba(features)[:, positive_index]
    scored = pd.DataFrame(
        {
            "client_id": features.index.get_level_values("client_id"),
            "family": features.index.get_level_values("family"),
            "score": values,
        }
    )
    client_order = pd.Index(features.index.get_level_values("client_id")).drop_duplicates()
    return (
        scored.groupby(["client_id", "family"], sort=False)["score"]
        .max()
        .unstack("family")
        .reindex(index=client_order, columns=FAMILIES, fill_value=0.0)
    )


def tune_pair_threshold(probabilities: pd.DataFrame, targets: pd.Series) -> tuple[float, float]:
    best = (-1.0, 0.5)
    aligned_targets = targets.loc[probabilities.index]
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = predict_pairs(probabilities, float(threshold))
        score = f1_score(aligned_targets, predicted, labels=LABELS, average="macro")
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1], best[0]


def tune_blend(
    multiclass: np.ndarray,
    pairs: pd.DataFrame,
    targets: pd.Series,
) -> tuple[float, float, float]:
    pair_array = pairs.to_numpy()
    actual = targets.loc[pairs.index].to_numpy()
    best = (-1.0, 0.0, 0.5)
    for weight in np.linspace(0.0, 1.0, 21):
        recurring = weight * multiclass[:, : len(FAMILIES)] + (1 - weight) * pair_array
        none_score = weight * multiclass[:, -1] + (1 - weight) * (1 - pair_array.max(axis=1))
        best_family = recurring.argmax(axis=1)
        margins = recurring[np.arange(len(recurring)), best_family] - none_score
        for threshold in np.linspace(-0.4, 0.4, 161):
            predicted = np.where(
                margins >= threshold,
                np.asarray(FAMILIES)[best_family],
                "none",
            )
            score = f1_score(actual, predicted, labels=LABELS, average="macro")
            if score > best[0]:
                best = (float(score), float(weight), float(threshold))
    return best[1], best[2], best[0]


def report(name: str, actual: pd.Series, predicted: np.ndarray) -> float:
    score = f1_score(actual, predicted, labels=LABELS, average="macro")
    print(f"\n{name}: macro_f1={score:.4f}")
    print(classification_report(actual, predicted, labels=LABELS, digits=3, zero_division=0))
    return float(score)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run challenge-focused forecasting experiments")
    parser.add_argument(
        "--feature-dir", type=Path, default=repository / "data" / "dataset_features"
    )
    parser.add_argument("--label-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_targets = load_targets(args.label_dir / "train_labels.csv")
    valid_targets = load_targets(args.label_dir / "valid_labels.csv")
    train = pd.read_csv(args.feature_dir / "train_features.csv")
    valid = pd.read_csv(args.feature_dir / "valid_features.csv")
    train_wide, train_pairs = build_feature_tables(train, train_targets.index)
    valid_wide, valid_pairs = build_feature_tables(valid, valid_targets.index)
    train_streams = build_stream_table(train, train_targets.index)
    valid_streams = build_stream_table(valid, valid_targets.index)

    train_wide = train_wide.reindex(columns=sorted(set(train_wide) | set(valid_wide)))
    valid_wide = valid_wide.reindex(columns=train_wide.columns)
    train_pairs = train_pairs.reindex(columns=sorted(set(train_pairs) | set(valid_pairs)))
    valid_pairs = valid_pairs.reindex(columns=train_pairs.columns)
    train_streams = train_streams.reindex(
        columns=sorted(set(train_streams) | set(valid_streams))
    )
    valid_streams = valid_streams.reindex(columns=train_streams.columns)

    multiclass = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True),
        ExtraTreesClassifier(
            n_estimators=900,
            min_samples_leaf=2,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=2026,
        ),
    )
    multiclass.fit(train_wide, train_targets.loc[train_wide.index])
    multiclass_probabilities = probabilities_in_label_order(multiclass, valid_wide)
    multiclass_predictions = np.asarray(LABELS)[multiclass_probabilities.argmax(axis=1)]
    multiclass_score = report(
        "stage 1 - multiclass recurrence features",
        valid_targets.loc[valid_wide.index],
        multiclass_predictions,
    )

    catboost_multiclass = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1400,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        auto_class_weights="Balanced",
        random_seed=2026,
        allow_writing_files=False,
        verbose=False,
    )
    catboost_multiclass.fit(
        train_wide,
        train_targets.loc[train_wide.index],
        eval_set=(valid_wide, valid_targets.loc[valid_wide.index]),
        early_stopping_rounds=120,
        verbose=False,
    )
    catboost_probabilities = probabilities_in_label_order(catboost_multiclass, valid_wide)
    catboost_predictions = np.asarray(LABELS)[catboost_probabilities.argmax(axis=1)]
    catboost_score = report(
        "stage 2 - CatBoost multiclass recurrence features",
        valid_targets.loc[valid_wide.index],
        catboost_predictions,
    )

    pair_targets = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_pairs.index],
        dtype=np.int8,
    )
    pair_model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True),
        HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=350,
            max_leaf_nodes=23,
            min_samples_leaf=18,
            l2_regularization=1.0,
            random_state=2026,
        ),
    )
    pair_weights = compute_sample_weight("balanced", pair_targets)
    pair_model.fit(train_pairs, pair_targets, histgradientboostingclassifier__sample_weight=pair_weights)
    valid_pair_probabilities = pair_probabilities(pair_model, valid_pairs)
    threshold, pair_score = tune_pair_threshold(valid_pair_probabilities, valid_targets)
    pair_predictions = predict_pairs(valid_pair_probabilities, threshold)
    report(
        f"stage 3 - shared family recurrence model (threshold={threshold:.3f})",
        valid_targets.loc[valid_pair_probabilities.index],
        pair_predictions,
    )

    stream_targets = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_streams.index],
        dtype=np.int8,
    )
    stream_model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=23,
            min_samples_leaf=25,
            l2_regularization=2.0,
            random_state=2026,
        ),
    )
    stream_weights = compute_sample_weight("balanced", stream_targets)
    stream_model.fit(
        train_streams,
        stream_targets,
        histgradientboostingclassifier__sample_weight=stream_weights,
    )
    valid_stream_probabilities = stream_probabilities(stream_model, valid_streams)
    stream_threshold, stream_score = tune_pair_threshold(
        valid_stream_probabilities, valid_targets
    )
    stream_predictions = predict_pairs(valid_stream_probabilities, stream_threshold)
    report(
        f"stage 4 - merchant stream model (threshold={stream_threshold:.3f})",
        valid_targets.loc[valid_stream_probabilities.index],
        stream_predictions,
    )

    blend_weight, blend_threshold, blend_score = tune_blend(
        catboost_probabilities, valid_pair_probabilities, valid_targets
    )
    recurring = (
        blend_weight * catboost_probabilities[:, : len(FAMILIES)]
        + (1 - blend_weight) * valid_pair_probabilities.to_numpy()
    )
    none_score = blend_weight * catboost_probabilities[:, -1] + (1 - blend_weight) * (
        1 - valid_pair_probabilities.max(axis=1).to_numpy()
    )
    family_index = recurring.argmax(axis=1)
    blend_predictions = np.where(
        recurring[np.arange(len(recurring)), family_index] - none_score >= blend_threshold,
        np.asarray(FAMILIES)[family_index],
        "none",
    )
    report(
        f"stage 5 - calibrated blend (weight={blend_weight:.2f}, threshold={blend_threshold:.3f})",
        valid_targets.loc[valid_pair_probabilities.index],
        blend_predictions,
    )

    bundle = {
        "catboost_multiclass_model": catboost_multiclass,
        "pair_model": pair_model,
        "wide_columns": train_wide.columns.tolist(),
        "pair_columns": train_pairs.columns.tolist(),
        "stream_columns": train_streams.columns.tolist(),
        "pair_threshold": threshold,
        "blend_weight": blend_weight,
        "blend_threshold": blend_threshold,
    }
    joblib.dump(bundle, args.output_dir / "model.joblib")
    metrics = {
        "multiclass_macro_f1": multiclass_score,
        "catboost_multiclass_macro_f1": catboost_score,
        "pair_macro_f1": pair_score,
        "stream_macro_f1": stream_score,
        "blend_macro_f1": blend_score,
        "pair_threshold": threshold,
        "blend_weight": blend_weight,
        "blend_threshold": blend_threshold,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "client_id": valid_pair_probabilities.index,
            "actual": valid_targets.loc[valid_pair_probabilities.index].to_numpy(),
            "predicted": blend_predictions,
        }
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    np.savez_compressed(
        args.output_dir / "tabular_valid_probabilities.npz",
        client_ids=np.asarray(valid_pair_probabilities.index, dtype=str),
        catboost=catboost_probabilities,
        pair=valid_pair_probabilities.to_numpy(),
        stream=valid_stream_probabilities.to_numpy(),
    )
    print(f"\nSaved experiment bundle and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
