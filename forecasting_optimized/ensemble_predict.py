"""Combine saved Fausto and noise-robust LightGBM probabilities without refitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from features import LABELS


def load_probabilities(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as saved:
        client_ids = saved["client_ids"].astype(str)
        probabilities = saved["probabilities"].astype(np.float64)
    if probabilities.shape != (len(client_ids), len(LABELS)):
        raise ValueError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    if len(set(client_ids)) != len(client_ids) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid client IDs or probabilities in {path}")
    return pd.DataFrame(probabilities, index=client_ids, columns=LABELS)


def predict(fausto: pd.DataFrame, lightgbm: pd.DataFrame, clients: pd.Index) -> np.ndarray:
    ids = clients.astype(str)
    missing = set(ids) - set(fausto.index) | (set(ids) - set(lightgbm.index))
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} clients")
    scores = 0.5 * fausto.loc[ids, LABELS].to_numpy() + 0.5 * lightgbm.loc[ids, LABELS].to_numpy()
    return np.asarray(LABELS)[scores.argmax(axis=1)]


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    artifacts = repository / "forecasting_optimized" / "artifacts"
    parser = argparse.ArgumentParser(description="Combine Fausto and LightGBM test probabilities")
    parser.add_argument("--fausto-valid", type=Path, default=artifacts / "blended_valid_probabilities.npz")
    parser.add_argument("--fausto-test", type=Path, default=artifacts / "test_probabilities.npz")
    parser.add_argument("--lightgbm-valid", type=Path, default=artifacts / "lightgbm" / "valid_probabilities.npz")
    parser.add_argument("--lightgbm-test", type=Path, default=artifacts / "lightgbm" / "test_probabilities.npz")
    parser.add_argument("--validation-labels", type=Path, default=repository / "data" / "dataset" / "valid_labels.csv")
    parser.add_argument("--template", type=Path, default=repository / "data" / "dataset" / "sample_submission.csv")
    parser.add_argument("--out", type=Path, default=repository / "submission.csv")
    args = parser.parse_args()

    labels = pd.read_csv(args.validation_labels, dtype={"client_id": str})
    valid_ids = pd.Index(labels["client_id"], name="client_id")
    if not valid_ids.is_unique:
        raise ValueError("Validation labels contain duplicate clients")
    actual = labels["target_next_recurring_merchant"].to_numpy()
    predicted_valid = predict(
        load_probabilities(args.fausto_valid),
        load_probabilities(args.lightgbm_valid),
        valid_ids,
    )
    score = f1_score(actual, predicted_valid, labels=LABELS, average="macro")
    print(f"Development validation macro-F1: {score:.4f}")
    print(classification_report(actual, predicted_valid, labels=LABELS, zero_division=0, digits=3))
    pd.DataFrame({"client_id": valid_ids, "actual": actual, "predicted": predicted_valid}).to_csv(
        artifacts / "ensemble_validation_predictions.csv", index=False
    )

    template = pd.read_csv(args.template, dtype={"client_id": str})
    if not template["client_id"].is_unique:
        raise ValueError("Submission template contains duplicate clients")
    template["predicted_next_recurring_merchant"] = predict(
        load_probabilities(args.fausto_test),
        load_probabilities(args.lightgbm_test),
        pd.Index(template["client_id"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.out, index=False)
    print(f"Wrote {len(template):,} test predictions to {args.out}")


if __name__ == "__main__":
    main()
