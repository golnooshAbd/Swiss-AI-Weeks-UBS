from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from features import FAMILIES, LABELS, build_feature_tables
from blend_models import as_distribution
from experiment import pair_probabilities, probabilities_in_label_order

def macro_f1_score(actual: np.ndarray, log_scores: np.ndarray, offsets: np.ndarray | None = None) -> float:
    scores = log_scores + offsets if offsets is not None else log_scores
    preds = scores.argmax(axis=1)
    return float(f1_score(actual, preds, average="macro", zero_division=0))

def tune_offsets(actual: np.ndarray, log_scores: np.ndarray) -> tuple[np.ndarray, float]:
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    best_score = macro_f1_score(actual, log_scores, offsets)
    for step in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for col in range(len(LABELS)):
                for direction in (-step, step):
                    candidate = offsets.copy()
                    candidate[col] += direction
                    score = macro_f1_score(actual, log_scores, candidate)
                    if score > best_score + 1e-7:
                        offsets, best_score = candidate, score
                        improved = True
    return offsets, best_score

def load_npz(path: Path, index_order, key="probabilities"):
    if not path.exists(): return None
    with np.load(path) as data:
        by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data[key])}
    return np.stack([by_client[cid] for cid in index_order])

def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    feature_dir = repository / "data" / "dataset_features"
    label_dir = repository / "data" / "dataset"
    artifact_dir = Path(__file__).parent / "artifacts"
    
    print("Loading labels...")
    train_labels = pd.read_csv(label_dir / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(label_dir / "valid_labels.csv", dtype={"client_id": str})
    train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
    valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]
    
    print("Building full feature tables (Wide & Pairs)...")
    train_wide, train_pairs = build_feature_tables(pd.read_csv(feature_dir / "train_features.csv"), train_targets.index)
    valid_wide, valid_pairs = build_feature_tables(pd.read_csv(feature_dir / "valid_features.csv"), valid_targets.index)
    
    train_wide = train_wide.fillna(0.0)
    valid_wide = valid_wide.reindex(columns=train_wide.columns).fillna(0.0)
    train_pairs = train_pairs.fillna(0.0)
    valid_pairs = valid_pairs.reindex(columns=train_pairs.columns).fillna(0.0)
    
    print(f"Wide shape: {train_wide.shape}, Pairs shape: {train_pairs.shape}")
    
    target_map = {name: idx for idx, name in enumerate(LABELS)}
    y_train_wide = train_targets.map(target_map).to_numpy()
    y_valid = valid_targets.map(target_map).to_numpy()
    
    train_binary = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_pairs.index],
        dtype=np.int8,
    )
    valid_binary = np.asarray(
        [valid_targets[client_id] == family for client_id, family in valid_pairs.index],
        dtype=np.int8,
    )
    sample_weights_pairs = compute_sample_weight("balanced", train_binary)
    sample_weights_wide = compute_sample_weight("balanced", y_train_wide)
    
    model_probs: dict[str, np.ndarray] = {}
    
    # --- 1. Train CatBoost Multiclass on Wide Table ---
    print("\n[1/4] Training CatBoost Multiclass on Wide...")
    cat_multi = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        early_stopping_rounds=80,
        random_seed=2026,
        verbose=False,
        thread_count=-1
    )
    cat_multi.fit(
        train_wide, y_train_wide,
        sample_weight=sample_weights_wide,
        eval_set=(valid_wide, y_valid),
        verbose=False
    )
    cat_multi_probs = cat_multi.predict_proba(valid_wide)
    # Ensure correct class mapping
    if hasattr(cat_multi, "classes_"):
        aligned = np.zeros_like(cat_multi_probs)
        for i, c in enumerate(cat_multi.classes_):
            aligned[:, c] = cat_multi_probs[:, i]
        cat_multi_probs = aligned
    model_probs["catboost_multi"] = cat_multi_probs
    offsets, score = tune_offsets(y_valid, np.log(np.clip(cat_multi_probs, 1e-7, 1.0)))
    print(f"CatBoost Multiclass Calibrated F1: {score:.5f}")
    
    # --- 2. Train XGBoost on Pairs ---
    print("\n[2/4] Training XGBoost Pair Model...")
    xgb_pair = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=8.0,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        tree_method="hist",
        early_stopping_rounds=100,
        random_state=2026,
        n_jobs=-1,
    )
    xgb_pair.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights_pairs,
        eval_set=[(valid_pairs, valid_binary)],
        verbose=False
    )
    xgb_rec = pair_probabilities(xgb_pair, valid_pairs).reindex(valid_targets.index).to_numpy()
    xgb_probs = as_distribution(xgb_rec)
    model_probs["xgboost_pair"] = xgb_probs
    offsets, score = tune_offsets(y_valid, np.log(np.clip(xgb_probs, 1e-7, 1.0)))
    print(f"XGBoost Pair Calibrated F1: {score:.5f}")
    
    # --- 3. Train LightGBM on Pairs ---
    print("\n[3/4] Training LightGBM Pair Model...")
    lgb_pair = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=24,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        random_state=2026,
        n_jobs=-1,
        verbose=-1
    )
    lgb_pair.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights_pairs,
        eval_set=[(valid_pairs, valid_binary)],
        callbacks=[]
    )
    lgb_rec = pair_probabilities(lgb_pair, valid_pairs).reindex(valid_targets.index).to_numpy()
    lgb_probs = as_distribution(lgb_rec)
    model_probs["lightgbm_pair"] = lgb_probs
    offsets, score = tune_offsets(y_valid, np.log(np.clip(lgb_probs, 1e-7, 1.0)))
    print(f"LightGBM Pair Calibrated F1: {score:.5f}")
    
    # --- 4. Train CatBoost on Pairs ---
    print("\n[4/4] Training CatBoost Pair Model...")
    cb_pair = CatBoostClassifier(
        iterations=1400,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        early_stopping_rounds=90,
        random_seed=2026,
        verbose=False,
        thread_count=-1
    )
    cb_pair.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights_pairs,
        eval_set=(valid_pairs, valid_binary),
        verbose=False
    )
    cb_rec = pair_probabilities(cb_pair, valid_pairs).reindex(valid_targets.index).to_numpy()
    cb_probs = as_distribution(cb_rec)
    model_probs["catboost_pair"] = cb_probs
    offsets, score = tune_offsets(y_valid, np.log(np.clip(cb_probs, 1e-7, 1.0)))
    print(f"CatBoost Pair Calibrated F1: {score:.5f}")
    
    # --- Load Existing Elite Artifacts ---
    print("\nLoading existing specialized models...")
    for name in ["stacking", "neural", "ranking", "embedding_pool", "description_stream", "family"]:
        p = artifact_dir / f"{name}_valid_probabilities.npz"
        probs = load_npz(p, valid_wide.index)
        if probs is not None:
            if probs.shape[1] == 7:
                probs = as_distribution(probs)
            model_probs[name] = probs
            print(f"Loaded {name:<20}: Shape {str(probs.shape):<10}")
            
    # Proxy models
    proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(v) for v in data["client_ids"]]
            positions = {cid: idx for idx, cid in enumerate(proxy_order)}
            model_probs["proxy_catboost"] = np.stack([data["catboost"][positions[cid]] for cid in valid_wide.index])
            model_probs["proxy_pair"] = np.stack([data["pair"][positions[cid]] for cid in valid_wide.index])
            print(f"Loaded proxy_catboost and proxy_pair")
            
    # Base bundle models from model.joblib (original baseline models)
    bundle = joblib.load(artifact_dir / "model.joblib")
    base_wide = valid_wide.reindex(columns=bundle["wide_columns"]).fillna(0.0)
    base_pairs = valid_pairs.reindex(columns=bundle["pair_columns"]).fillna(0.0)
    model_probs["orig_catboost"] = probabilities_in_label_order(bundle["catboost_multiclass_model"], base_wide)
    model_probs["orig_pair"] = as_distribution(pair_probabilities(bundle["pair_model"], base_pairs).to_numpy())
    print("Loaded original base CatBoost and Pair models")

    # --- Hyper-Optimization of Ensembles ---
    print(f"\n==================================================")
    print(f"Total Candidate Probability Engines: {len(model_probs)}")
    print(f"Engines: {list(model_probs.keys())}")
    print(f"==================================================")

    # Systematic Blend Optimization via Multi-Start Coordinate Descent
    import random
    
    best_blend_score = 0.60847  # Must beat baseline
    best_blend_weights = None
    best_blend_offsets = None
    
    # We test linear and geometric combinations
    models_list = list(model_probs.keys())
    
    print("\nRunning 10,000 iterations of high-dimensional ensemble optimization...")
    
    for iteration in range(10000):
        # Sample active models and weights (Dirichlet distribution over a subset)
        k = random.randint(4, len(models_list))
        selected_models = random.sample(models_list, k)
        raw_w = np.random.exponential(scale=1.0, size=k)
        weights = dict(zip(selected_models, (raw_w / raw_w.sum()).tolist()))
        
        # Linear or geometric blending
        is_geometric = random.random() < 0.7
        if is_geometric:
            log_blend = np.zeros((len(valid_wide), len(LABELS)), dtype=np.float64)
            for m, w in weights.items():
                log_blend += w * np.log(np.clip(model_probs[m], 1e-7, 1.0))
        else:
            blend = np.zeros((len(valid_wide), len(LABELS)), dtype=np.float64)
            for m, w in weights.items():
                blend += w * model_probs[m]
            log_blend = np.log(np.clip(blend, 1e-7, 1.0))
            
        candidate_offsets, candidate_score = tune_offsets(y_valid, log_blend)
        
        if candidate_score > best_blend_score:
            best_blend_score = candidate_score
            best_blend_weights = weights
            best_blend_offsets = candidate_offsets
            best_is_geometric = is_geometric
            print(f"Iter {iteration:5d} | NEW RECORD MACRO-F1: {best_blend_score:.5f} | Mode: {'Geometric' if is_geometric else 'Linear'} | Models: {len(weights)}")
            
    print(f"\n==================================================")
    print(f"OPTIMIZATION FINISHED. HIGHEST SCORE ACHIEVED: {best_blend_score:.5f}")
    print(f"==================================================")
    
    if best_blend_weights is not None:
        print("Winning Weights:")
        for m, w in sorted(best_blend_weights.items(), key=lambda x: -x[1]):
            print(f"  {m:<25}: {w:.4f}")
        print("\nWinning Offsets:")
        for l, off in zip(LABELS, best_blend_offsets):
            print(f"  {l:<12}: {off:+.4f}")
            
        # Compute final report
        if best_is_geometric:
            log_final = np.zeros((len(valid_wide), len(LABELS)), dtype=np.float64)
            for m, w in best_blend_weights.items():
                log_final += w * np.log(np.clip(model_probs[m], 1e-7, 1.0))
        else:
            final = np.zeros((len(valid_wide), len(LABELS)), dtype=np.float64)
            for m, w in best_blend_weights.items():
                final += w * model_probs[m]
            log_final = np.log(np.clip(final, 1e-7, 1.0))
            
        preds = (log_final + best_blend_offsets).argmax(axis=1)
        print("\nDetailed Per-Class Performance:")
        print(classification_report(y_valid, preds, target_names=LABELS, digits=5))
        
        # Save models and config
        joblib.dump(cat_multi, artifact_dir / "catboost_super_multiclass.joblib")
        joblib.dump(xgb_pair, artifact_dir / "xgboost_super_pair.joblib")
        joblib.dump(lgb_pair, artifact_dir / "lightgbm_super_pair.joblib")
        joblib.dump(cb_pair, artifact_dir / "catboost_super_pair.joblib")
        
        np.savez_compressed(artifact_dir / "catboost_super_valid_probabilities.npz", client_ids=valid_wide.index, probabilities=cat_multi_probs)
        np.savez_compressed(artifact_dir / "xgboost_super_valid_probabilities.npz", client_ids=valid_wide.index, probabilities=xgb_probs)
        np.savez_compressed(artifact_dir / "lightgbm_super_valid_probabilities.npz", client_ids=valid_wide.index, probabilities=lgb_probs)
        np.savez_compressed(artifact_dir / "catboost_pair_super_valid_probabilities.npz", client_ids=valid_wide.index, probabilities=cb_probs)
        
        winning_config = {
            "macro_f1": float(best_blend_score),
            "is_geometric": bool(best_is_geometric),
            "weights": {k: float(v) for k, v in best_blend_weights.items()},
            "class_offsets": {l: float(off) for l, off in zip(LABELS, best_blend_offsets)}
        }
        with open(artifact_dir / "super_blend_config.json", "w") as f:
            json.dump(winning_config, f, indent=2)
        print("Saved super_blend_config.json and all model weights!")

if __name__ == "__main__":
    main()
