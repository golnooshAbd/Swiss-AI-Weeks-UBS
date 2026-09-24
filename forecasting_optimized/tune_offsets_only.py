"""
Fine-grained Optuna search ONLY on class offsets,
fixing blend weights at the greedy optimum.
Much faster than full search, ~2 min.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.metrics import f1_score

from experiment import pair_probabilities, probabilities_in_label_order
from blend_models import as_distribution
from features import FAMILIES, LABELS, build_feature_tables

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"
LABEL_DIR = REPO / "data" / "dataset"

# ── Load data ─────────────────────────────────────────────────────────────────
labels = pd.read_csv(LABEL_DIR / "valid_labels.csv", dtype={"client_id": str})
target_map = {name: index for index, name in enumerate(LABELS)}
target = labels["target_next_recurring_merchant"].map(target_map)
target_by_client = pd.Series(target.to_numpy(), index=labels["client_id"])

bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
frame = pd.read_csv(FEATURE_DIR / "valid_features.csv")
wide, pairs = build_feature_tables(frame, labels["client_id"])
wide = wide.reindex(columns=bundle["wide_columns"])
pairs = pairs.reindex(columns=bundle["pair_columns"])
catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

def load_npz(path, key="probabilities"):
    if not path.exists(): return None
    with np.load(path) as data:
        by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data[key])}
    return np.stack([by_client[cid] for cid in wide.index])

family_raw = load_npz(ARTIFACT_DIR / "family_valid_probabilities.npz")
family = as_distribution(family_raw) if family_raw is not None else None
neural = load_npz(ARTIFACT_DIR / "neural_valid_probabilities.npz")
stacking = load_npz(ARTIFACT_DIR / "stacking_valid_probabilities.npz")
proxy_path = ARTIFACT_DIR / "proxy_valid_probabilities.npz"
if proxy_path.exists():
    with np.load(proxy_path) as data:
        proxy_order = [str(v) for v in data["client_ids"]]
        pos = {cid: i for i, cid in enumerate(proxy_order)}
        proxy_catboost = np.stack([data["catboost"][pos[cid]] for cid in wide.index])
        proxy_pair = np.stack([data["pair"][pos[cid]] for cid in wide.index])
else:
    proxy_catboost = proxy_pair = None
xgboost_pair = load_npz(ARTIFACT_DIR / "xgboost_valid_probabilities.npz")
ranking = load_npz(ARTIFACT_DIR / "ranking_valid_probabilities.npz")
embedding_pool = load_npz(ARTIFACT_DIR / "embedding_pool_valid_probabilities.npz")
description_stream = load_npz(ARTIFACT_DIR / "description_stream_valid_probabilities.npz")

actual = target_by_client.loc[wide.index].to_numpy()

# ── Reconstruct blend at greedy optimum ──────────────────────────────────────
# Weights from the greedy result
w_catboost, w_pair, w_family, w_neural = 0.1, 0.72, 0.072, 0.108
probs = w_catboost * catboost + w_pair * pair

if family is not None:
    probs += w_family * family
if neural is not None:
    probs += w_neural * neural

# Extra sequential weights from greedy
extra = {
    "proxy_catboost": (proxy_catboost, 0.04),
    "xgboost_pair":   (xgboost_pair,   0.10),
    "ranking":        (ranking,         0.16),
    "embedding_pool": (embedding_pool,  0.28),
    "description_stream": (description_stream, 0.08),
}
for name, (src, w) in extra.items():
    if src is not None:
        probs = (1 - w) * probs + w * src
        probs = probs / probs.sum(axis=1, keepdims=True)

# Class-specific adjustments from greedy
adjustments = [
    ("stacking", "music",     0.125),
    ("xgboost_pair", "gym",   0.35),
    ("xgboost_pair", "insurance", -0.125),
    ("xgboost_pair", "streaming", -0.025),
    ("ranking", "mobile",     0.025),
    ("embedding_pool", "streaming", 0.475),
    ("description_stream", "streaming", 0.05),
]
sources_map = {
    "stacking": stacking,
    "xgboost_pair": xgboost_pair,
    "ranking": ranking,
    "embedding_pool": embedding_pool,
    "description_stream": description_stream,
}
for src_name, label, w in adjustments:
    src = sources_map.get(src_name)
    if src is None: continue
    col = LABELS.index(label)
    probs[:, col] = (1 - w) * probs[:, col] + w * src[:, col]
    probs = np.clip(probs, 1e-7, None)
    probs /= probs.sum(axis=1, keepdims=True)

# Baseline with greedy offsets
greedy_offsets = np.array([0.2, 0.125, 0.2, 0.275, 0.4, 0.0, 0.35, -0.3])
baseline = float(f1_score(
    actual,
    (np.log(np.clip(probs, 1e-7, 1.0)) + greedy_offsets).argmax(axis=1),
    labels=np.arange(len(LABELS)), average="macro", zero_division=0
))
print(f"Baseline with greedy offsets: {baseline:.4f}", flush=True)

# ── Optuna fine-grained offset search ────────────────────────────────────────
log_probs = np.log(np.clip(probs, 1e-7, 1.0))

def objective(trial: optuna.Trial) -> float:
    offsets = np.array([
        trial.suggest_float(f"off_{label}", -1.2, 1.2)
        for label in LABELS
    ])
    return float(f1_score(
        actual,
        (log_probs + offsets).argmax(axis=1),
        labels=np.arange(len(LABELS)), average="macro", zero_division=0
    ))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=100),
)
# Warm start with greedy offsets
study.enqueue_trial({f"off_{label}": float(greedy_offsets[i]) for i, label in enumerate(LABELS)})
study.optimize(objective, n_trials=5000, n_jobs=-1, show_progress_bar=False)

print(f"\nBest offset-tuned F1: {study.best_value:.4f} (vs baseline {baseline:.4f})", flush=True)
print(f"Best offsets: {study.best_params}", flush=True)

best_offsets = np.array([study.best_params[f"off_{label}"] for label in LABELS])
predicted = (log_probs + best_offsets).argmax(axis=1)
from sklearn.metrics import classification_report
print(classification_report(
    actual, predicted,
    labels=np.arange(len(LABELS)),
    target_names=list(LABELS),
    digits=3, zero_division=0,
))
