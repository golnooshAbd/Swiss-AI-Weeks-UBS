"""
Train CatBoost with new features.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

from features import FAMILIES, LABELS, build_feature_tables
from experiment import probabilities_in_label_order
from blend_models import tune_offsets
import joblib

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"
LABEL_DIR = REPO / "data" / "dataset"

print("Loading and building new features from raw transactions...", flush=True)
train_labels = pd.read_csv(LABEL_DIR / "train_labels.csv", dtype={"client_id": str})
valid_labels = pd.read_csv(LABEL_DIR / "valid_labels.csv", dtype={"client_id": str})
train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]

train_frame = pd.read_csv(FEATURE_DIR / "train_features.csv")
valid_frame = pd.read_csv(FEATURE_DIR / "valid_features.csv")

# This will now include our new features automatically
train_wide, _ = build_feature_tables(train_frame, train_targets.index)
valid_wide, _ = build_feature_tables(valid_frame, valid_targets.index)

train_wide = train_wide.reindex(columns=sorted(set(train_wide) | set(valid_wide)))
valid_wide = valid_wide.reindex(columns=train_wide.columns)

print("Extracting TF-IDF text features via SVD...", flush=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def get_text(frame):
    frame["clean_description"] = frame["clean_description"].fillna("").astype(str)
    return frame.groupby("client_id")["clean_description"].apply(lambda texts: " ".join(texts))

train_text = get_text(train_frame).reindex(train_targets.index, fill_value="")
valid_text = get_text(valid_frame).reindex(valid_targets.index, fill_value="")

tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True)
train_tfidf = tfidf.fit_transform(train_text)
valid_tfidf = tfidf.transform(valid_text)

svd = TruncatedSVD(n_components=150, random_state=2026)
train_svd = svd.fit_transform(train_tfidf)
valid_svd = svd.transform(valid_tfidf)

for i in range(150):
    train_wide[f"tfidf_svd_{i}"] = pd.Series(train_svd[:, i], index=train_targets.index)
    valid_wide[f"tfidf_svd_{i}"] = pd.Series(valid_svd[:, i], index=valid_targets.index)

target_map = {name: index for index, name in enumerate(LABELS)}
actual = valid_targets.map(target_map).to_numpy()

print(f"Data ready. Wide shape: {train_wide.shape}", flush=True)
print("Training CatBoost with new features...", flush=True)

model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=2500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5.0,
    random_strength=0.5,
    bagging_temperature=0.2,
    border_count=128,
    auto_class_weights="Balanced",
    random_seed=2026,
    allow_writing_files=False,
    verbose=False,
    task_type="CPU"
)
model.fit(
    train_wide,
    train_targets.loc[train_wide.index],
    eval_set=(valid_wide, valid_targets.loc[valid_wide.index]),
    early_stopping_rounds=120,
    verbose=False,
)

probs = probabilities_in_label_order(model, valid_wide)
_, score = tune_offsets(actual, probs)

print(f"New feature CatBoost standalone F1: {score:.4f}", flush=True)

np.savez_compressed(
    ARTIFACT_DIR / "catboost_tuned_valid_probabilities.npz",
    client_ids=np.asarray(valid_wide.index, dtype=str),
    probabilities=probs,
)
print("Saved catboost_tuned_valid_probabilities.npz")

joblib.dump(
    {
        "model": model,
        "tfidf": tfidf,
        "svd": svd
    },
    ARTIFACT_DIR / "catboost_tuned_bundle.joblib"
)
print("Saved catboost_tuned_bundle.joblib")
