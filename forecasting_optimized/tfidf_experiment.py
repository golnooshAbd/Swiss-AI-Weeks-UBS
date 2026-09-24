"""
TF-IDF + Logistic Regression model on client transaction descriptions.
Extracts an orthogonal text signal to differentiate highly confused digital subscriptions
(music vs streaming vs software).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from blend_models import as_distribution, tune_offsets
from features import LABELS

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"
LABEL_DIR = REPO / "data" / "dataset"


def aggregate_text(path: Path) -> pd.Series:
    """Concatenate all transaction descriptions for each client into a single document."""
    frame = pd.read_csv(path, usecols=["client_id", "clean_description"])
    frame["clean_description"] = frame["clean_description"].fillna("").astype(str)
    return frame.groupby("client_id")["clean_description"].apply(lambda texts: " ".join(texts))


def main() -> None:
    print("Loading data...", flush=True)
    train_labels = pd.read_csv(LABEL_DIR / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(LABEL_DIR / "valid_labels.csv", dtype={"client_id": str})
    
    train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
    valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]
    
    target_map = {name: index for index, name in enumerate(LABELS)}
    y_train = train_targets.map(target_map).to_numpy()
    y_valid = valid_targets.map(target_map).to_numpy()

    print("Aggregating text descriptions per client...", flush=True)
    X_train_series = aggregate_text(FEATURE_DIR / "train_features.csv").reindex(train_targets.index, fill_value="")
    X_valid_series = aggregate_text(FEATURE_DIR / "valid_features.csv").reindex(valid_targets.index, fill_value="")
    
    print("Training TF-IDF + Logistic Regression...", flush=True)
    model = make_pipeline(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=2026,
            n_jobs=-1
        )
    )
    
    model.fit(X_train_series, y_train)
    
    raw_probs = model.predict_proba(X_valid_series)
    # Ensure all classes are present
    probs = np.zeros((len(X_valid_series), len(LABELS)), dtype=np.float64)
    for i, cls in enumerate(model.classes_):
        probs[:, cls] = raw_probs[:, i]
        
    probs = np.clip(probs, 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    
    offsets, score = tune_offsets(y_valid, probs)
    predicted = (np.log(np.clip(probs, 1e-7, 1.0)) + offsets).argmax(axis=1)
    
    print(f"TF-IDF model standalone macro_f1={score:.4f}", flush=True)
    print(classification_report(
        y_valid,
        predicted,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        digits=3,
        zero_division=0,
    ))
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "offsets": offsets}, ARTIFACT_DIR / "tfidf_model.joblib")
    
    np.savez_compressed(
        ARTIFACT_DIR / "tfidf_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        probabilities=probs,
    )
    print("Saved tfidf_valid_probabilities.npz", flush=True)


if __name__ == "__main__":
    main()
