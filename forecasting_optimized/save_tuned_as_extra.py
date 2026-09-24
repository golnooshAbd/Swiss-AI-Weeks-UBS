"""
Save tuned CatBoost and Pair predictions as extra NPZ sources
WITHOUT touching model.joblib (keeps all existing NPZs consistent).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight

from experiment import pair_probabilities, probabilities_in_label_order
from blend_models import as_distribution, tune_offsets
from features import FAMILIES, LABELS, build_feature_tables

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"
LABEL_DIR = REPO / "data" / "dataset"

print("Loading data...", flush=True)
train_labels = pd.read_csv(LABEL_DIR / "train_labels.csv", dtype={"client_id": str})
valid_labels = pd.read_csv(LABEL_DIR / "valid_labels.csv", dtype={"client_id": str})
train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]

train_frame = pd.read_csv(FEATURE_DIR / "train_features.csv")
valid_frame = pd.read_csv(FEATURE_DIR / "valid_features.csv")
train_wide, train_pairs = build_feature_tables(train_frame, train_targets.index)
valid_wide, valid_pairs = build_feature_tables(valid_frame, valid_targets.index)

train_wide = train_wide.reindex(columns=sorted(set(train_wide) | set(valid_wide)))
valid_wide = valid_wide.reindex(columns=train_wide.columns)
train_pairs = train_pairs.reindex(columns=sorted(set(train_pairs) | set(valid_pairs)))
valid_pairs = valid_pairs.reindex(columns=train_pairs.columns)

target_map = {name: index for index, name in enumerate(LABELS)}
actual = valid_targets.map(target_map).to_numpy()

pair_targets_arr = np.asarray(
    [train_targets[cid] == fam for cid, fam in train_pairs.index], dtype=np.int8
)
pair_weights = compute_sample_weight("balanced", pair_targets_arr)
print("Data loaded.", flush=True)

# ── Train BEST CatBoost (from tuning results) ─────────────────────────────────
print("\nTraining best CatBoost...", flush=True)
best_catboost = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=1400,
    learning_rate=0.04047429096227562,
    depth=5,
    l2_leaf_reg=14.685882775837385,
    random_strength=0.5772366140454133,
    bagging_temperature=0.24609825069437713,
    border_count=64,
    auto_class_weights="Balanced",
    random_seed=2026,
    allow_writing_files=False,
    verbose=False,
    early_stopping_rounds=120,
)
best_catboost.fit(
    train_wide,
    train_targets.loc[train_wide.index],
    eval_set=(valid_wide, valid_targets.loc[valid_wide.index]),
    verbose=False,
)
catboost_probs = probabilities_in_label_order(best_catboost, valid_wide)
_, catboost_score = tune_offsets(actual, catboost_probs)
print(f"Tuned CatBoost standalone F1: {catboost_score:.4f}", flush=True)

# Save as extra source
np.savez_compressed(
    ARTIFACT_DIR / "catboost_tuned_valid_probabilities.npz",
    client_ids=np.asarray(valid_wide.index, dtype=str),
    probabilities=catboost_probs,
)
print("Saved catboost_tuned_valid_probabilities.npz", flush=True)

# ── Train BEST Pair model (from tuning results) ───────────────────────────────
print("\nTraining best Pair model...", flush=True)
best_pair = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True),
    HistGradientBoostingClassifier(
        learning_rate=0.05512873768544145,
        max_iter=300,
        max_leaf_nodes=58,
        min_samples_leaf=28,
        l2_regularization=3.6946510568577344,
        random_state=2026,
    ),
)
best_pair.fit(
    train_pairs,
    pair_targets_arr,
    histgradientboostingclassifier__sample_weight=pair_weights,
)
raw_pair = pair_probabilities(best_pair, valid_pairs)
pair_dist = as_distribution(raw_pair.reindex(valid_targets.index).to_numpy())
_, pair_score = tune_offsets(actual, pair_dist)
print(f"Tuned Pair standalone F1: {pair_score:.4f}", flush=True)

# Save as extra source
np.savez_compressed(
    ARTIFACT_DIR / "pair_tuned_valid_probabilities.npz",
    client_ids=np.asarray(valid_targets.index, dtype=str),
    probabilities=pair_dist,
)
print("Saved pair_tuned_valid_probabilities.npz", flush=True)
print("\nDone. Now run blend_models_v2.py", flush=True)
