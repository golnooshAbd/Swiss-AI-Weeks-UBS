"""
Optuna hyperparameter search for CatBoost + HistGBM pair model.
Saves the best models and their .npz predictions for re-blending.
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
from features import FAMILIES, LABELS, build_feature_tables, build_stream_table

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"
LABEL_DIR = REPO / "data" / "dataset"

# ── Load data once ────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_labels = pd.read_csv(LABEL_DIR / "train_labels.csv", dtype={"client_id": str})
valid_labels = pd.read_csv(LABEL_DIR / "valid_labels.csv", dtype={"client_id": str})
train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]

train_frame = pd.read_csv(FEATURE_DIR / "train_features.csv")
valid_frame = pd.read_csv(FEATURE_DIR / "valid_features.csv")
train_wide, train_pairs = build_feature_tables(train_frame, train_targets.index)
valid_wide, valid_pairs = build_feature_tables(valid_frame, valid_targets.index)

# align columns
train_wide = train_wide.reindex(columns=sorted(set(train_wide) | set(valid_wide)))
valid_wide = valid_wide.reindex(columns=train_wide.columns)
train_pairs = train_pairs.reindex(columns=sorted(set(train_pairs) | set(valid_pairs)))
valid_pairs = valid_pairs.reindex(columns=train_pairs.columns)

target_map = {name: index for index, name in enumerate(LABELS)}
actual = valid_targets.map(target_map).to_numpy()

pair_targets = np.asarray(
    [train_targets[cid] == fam for cid, fam in train_pairs.index], dtype=np.int8
)
pair_weights = compute_sample_weight("balanced", pair_targets)

print("Data loaded.", flush=True)
print(f"train_wide: {train_wide.shape}, valid_wide: {valid_wide.shape}", flush=True)
print(f"train_pairs: {train_pairs.shape}", flush=True)

# ── Best trackers ─────────────────────────────────────────────────────────────
best_catboost_score = 0.0
best_pair_score = 0.0
best_catboost_model = None
best_pair_model = None
best_catboost_probs = None
best_pair_probs = None


def objective_catboost(trial: optuna.Trial) -> float:
    global best_catboost_score, best_catboost_model, best_catboost_probs
    iterations = trial.suggest_int("iterations", 800, 2200, step=200)
    depth = trial.suggest_int("depth", 5, 9)
    lr = trial.suggest_float("learning_rate", 0.02, 0.08, log=True)
    l2 = trial.suggest_float("l2_leaf_reg", 1.0, 15.0)
    rs = trial.suggest_float("random_strength", 0.1, 2.0)
    bagging = trial.suggest_float("bagging_temperature", 0.0, 1.5)
    border_count = trial.suggest_categorical("border_count", [64, 128, 256])

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        learning_rate=lr,
        depth=depth,
        l2_leaf_reg=l2,
        random_strength=rs,
        bagging_temperature=bagging,
        border_count=border_count,
        auto_class_weights="Balanced",
        random_seed=2026,
        allow_writing_files=False,
        verbose=False,
        early_stopping_rounds=120,
    )
    model.fit(
        train_wide,
        train_targets.loc[train_wide.index],
        eval_set=(valid_wide, valid_targets.loc[valid_wide.index]),
        verbose=False,
    )
    probs = probabilities_in_label_order(model, valid_wide)
    offsets, score = tune_offsets(actual, probs)
    print(f"  catboost trial score={score:.4f} depth={depth} lr={lr:.4f} l2={l2:.2f}", flush=True)
    if score > best_catboost_score:
        best_catboost_score = score
        best_catboost_model = model
        best_catboost_probs = probs
        print(f"  *** NEW BEST CatBoost: {score:.4f} ***", flush=True)
    return score


def objective_pair(trial: optuna.Trial) -> float:
    global best_pair_score, best_pair_model, best_pair_probs
    lr = trial.suggest_float("learning_rate", 0.02, 0.12, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 600, step=50)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 8, 40)
    l2 = trial.suggest_float("l2_regularization", 0.1, 10.0, log=True)

    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1.0, add_indicator=True),
        HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2,
            random_state=2026,
        ),
    )
    model.fit(
        train_pairs,
        pair_targets,
        histgradientboostingclassifier__sample_weight=pair_weights,
    )
    raw = pair_probabilities(model, valid_pairs)
    distribution = as_distribution(raw.reindex(valid_targets.index).to_numpy())
    offsets, score = tune_offsets(actual, distribution)
    print(f"  pair trial score={score:.4f} lr={lr:.4f} leaves={max_leaf_nodes} min_leaf={min_samples_leaf}", flush=True)
    if score > best_pair_score:
        best_pair_score = score
        best_pair_model = model
        best_pair_probs = distribution
        print(f"  *** NEW BEST Pair: {score:.4f} ***", flush=True)
    return score


# ── Run studies ───────────────────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("\n=== Tuning CatBoost ===", flush=True)
catboost_study = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42))
catboost_study.optimize(objective_catboost, n_trials=40, n_jobs=1, show_progress_bar=False)
print(f"\nBest CatBoost F1: {catboost_study.best_value:.4f}", flush=True)
print(f"Best CatBoost params: {catboost_study.best_params}", flush=True)

print("\n=== Tuning Pair Model ===", flush=True)
pair_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
pair_study.optimize(objective_pair, n_trials=40, n_jobs=1, show_progress_bar=False)
print(f"\nBest Pair F1: {pair_study.best_value:.4f}", flush=True)
print(f"Best Pair params: {pair_study.best_params}", flush=True)

# ── Save best models and probabilities ────────────────────────────────────────
print("\nSaving best models...", flush=True)

if best_catboost_model is not None:
    bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
    bundle["catboost_multiclass_model"] = best_catboost_model
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")
    print(f"Saved tuned CatBoost (F1={best_catboost_score:.4f})", flush=True)

if best_pair_model is not None:
    bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
    bundle["pair_model"] = best_pair_model
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")
    # Also save npz so blend_models can pick it up (overwrite xgboost slot to not break blend)
    np.savez_compressed(
        ARTIFACT_DIR / "pair_tuned_valid_probabilities.npz",
        client_ids=np.asarray(valid_targets.index, dtype=str),
        probabilities=best_pair_probs,
    )
    print(f"Saved tuned Pair model (F1={best_pair_score:.4f})", flush=True)

print("\nDone. Run blend_models.py to re-blend with tuned models.", flush=True)
