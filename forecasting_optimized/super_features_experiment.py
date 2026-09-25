import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

from features import FAMILIES, LABELS, build_feature_tables
from blend_models import as_distribution, tune_offsets
from experiment import pair_probabilities

def add_temporal_future_signals(pairs_df: pd.DataFrame) -> pd.DataFrame:
    df = pairs_df.copy()
    
    # We group by client_id (level 0 of multiindex)
    results = []
    
    # Reset index to easily manipulate client_id and family
    df_reset = df.reset_index()
    
    # Add new feature columns initialized
    new_cols = {
        "proj_rank": [],
        "is_proj_rank_1": [],
        "is_proj_rank_2": [],
        "proj_gap_to_min": [],
        "is_lapsed": [],
        "is_active_candidate": [],
        "active_candidate_count": [],
        "is_only_active_candidate": [],
        "amount_below_16": [],
        "amount_16_to_28": [],
        "amount_above_28": []
    }
    
    for client_id, group in df_reset.groupby("client_id", sort=False):
        # Projected days for each row
        proj_days_map = {}
        cycles_map = {}
        counts_map = {}
        amt_med_map = {}
        
        for _, row in group.iterrows():
            fam = row["family"]
            p_days = row.get("projected_days_to_next", np.nan)
            c_since = row.get("cycles_since_last", np.nan)
            cnt = row.get("count", 0.0)
            amed = row.get("amount_median", 0.0)
            
            proj_days_map[fam] = p_days
            cycles_map[fam] = c_since
            counts_map[fam] = cnt
            amt_med_map[fam] = amed
            
        valid_projs = {
            f: p for f, p in proj_days_map.items() 
            if np.isfinite(p) and 0 <= p <= 90
        }
        min_p = min(valid_projs.values()) if valid_projs else np.nan
        
        # Determine active candidates: count >= 2, cycles <= 1.8, proj in [0, 90]
        active_fams = [
            f for f in group["family"]
            if counts_map[f] >= 2 and np.isfinite(cycles_map[f]) and cycles_map[f] <= 1.8 and f in valid_projs
        ]
        n_active = float(len(active_fams))
        
        for _, row in group.iterrows():
            fam = row["family"]
            p_val = proj_days_map[fam]
            c_val = cycles_map[fam]
            a_med = amt_med_map[fam]
            
            if np.isfinite(p_val) and 0 <= p_val <= 90:
                rank = float(1 + sum(v < p_val for v in valid_projs.values()))
                gap = float(p_val - min_p)
            else:
                rank = 8.0
                gap = 999.0
                
            is_lapsed_val = float(np.isfinite(c_val) and c_val > 1.8)
            is_active_val = float(fam in active_fams)
            
            new_cols["proj_rank"].append(rank)
            new_cols["is_proj_rank_1"].append(float(rank == 1.0))
            new_cols["is_proj_rank_2"].append(float(rank == 2.0))
            new_cols["proj_gap_to_min"].append(gap)
            new_cols["is_lapsed"].append(is_lapsed_val)
            new_cols["is_active_candidate"].append(is_active_val)
            new_cols["active_candidate_count"].append(n_active)
            new_cols["is_only_active_candidate"].append(float(n_active == 1.0 and is_active_val))
            
            new_cols["amount_below_16"].append(float(a_med > 0 and a_med < 16.0))
            new_cols["amount_16_to_28"].append(float(16.0 <= a_med < 28.0))
            new_cols["amount_above_28"].append(float(a_med >= 28.0))
            
    for col, vals in new_cols.items():
        df_reset[col] = vals
        
    return df_reset.set_index(["client_id", "family"]).sort_index(axis=1)

def main():
    repository = Path(__file__).resolve().parents[1]
    feature_dir = repository / "data" / "dataset_features"
    label_dir = repository / "data" / "dataset"
    
    train_labels = pd.read_csv(label_dir / "train_labels.csv", dtype={"client_id": str})
    valid_labels = pd.read_csv(label_dir / "valid_labels.csv", dtype={"client_id": str})
    train_targets = train_labels.set_index("client_id")["target_next_recurring_merchant"]
    valid_targets = valid_labels.set_index("client_id")["target_next_recurring_merchant"]
    
    print("Building feature tables...")
    _, train_pairs = build_feature_tables(pd.read_csv(feature_dir / "train_features.csv"), train_targets.index)
    _, valid_pairs = build_feature_tables(pd.read_csv(feature_dir / "valid_features.csv"), valid_targets.index)
    
    print("Adding super temporal & pricing signals...")
    train_pairs = add_temporal_future_signals(train_pairs).fillna(0.0)
    valid_pairs = add_temporal_future_signals(valid_pairs).fillna(0.0)
    
    train_binary = np.asarray(
        [train_targets[client_id] == family for client_id, family in train_pairs.index],
        dtype=np.int8,
    )
    valid_binary = np.asarray(
        [valid_targets[client_id] == family for client_id, family in valid_pairs.index],
        dtype=np.int8,
    )
    sample_weights = compute_sample_weight("balanced", train_binary)
    actual = valid_targets.map({label: index for index, label in enumerate(LABELS)}).to_numpy()
    
    print(f"Total features in pairs: {train_pairs.shape[1]}")
    
    # 1. XGBoost Pair
    print("\n--- Training XGBoost Pair Model ---")
    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10.0,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        tree_method="hist",
        early_stopping_rounds=120,
        random_state=2026,
        n_jobs=-1,
    )
    xgb.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights,
        eval_set=[(valid_pairs, valid_binary)],
        verbose=False
    )
    xgb_rec = pair_probabilities(xgb, valid_pairs).reindex(valid_targets.index).to_numpy()
    xgb_dist = as_distribution(xgb_rec)
    xgb_offsets, xgb_score = tune_offsets(actual, xgb_dist)
    print(f"XGBoost Standalone Calibrated Macro-F1: {xgb_score:.5f}")
    
    # 2. LightGBM Pair
    print("\n--- Training LightGBM Pair Model ---")
    lgb = LGBMClassifier(
        n_estimators=1800,
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
    lgb.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights,
        eval_set=[(valid_pairs, valid_binary)],
        callbacks=[]
    )
    lgb_rec = pair_probabilities(lgb, valid_pairs).reindex(valid_targets.index).to_numpy()
    lgb_dist = as_distribution(lgb_rec)
    lgb_offsets, lgb_score = tune_offsets(actual, lgb_dist)
    print(f"LightGBM Standalone Calibrated Macro-F1: {lgb_score:.5f}")
    
    # 3. CatBoost Pair
    print("\n--- Training CatBoost Pair Model ---")
    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        early_stopping_rounds=100,
        random_seed=2026,
        verbose=False,
        thread_count=-1
    )
    cb.fit(
        train_pairs, train_binary,
        sample_weight=sample_weights,
        eval_set=(valid_pairs, valid_binary),
        verbose=False
    )
    cb_rec = pair_probabilities(cb, valid_pairs).reindex(valid_targets.index).to_numpy()
    cb_dist = as_distribution(cb_rec)
    cb_offsets, cb_score = tune_offsets(actual, cb_dist)
    print(f"CatBoost Standalone Calibrated Macro-F1: {cb_score:.5f}")
    
    # 4. Trinity Pair Blend
    print("\n--- Trinity Pair Ensemble (XGBoost + LightGBM + CatBoost) ---")
    trinity_dist = (xgb_dist + lgb_dist + cb_dist) / 3.0
    trinity_offsets, trinity_score = tune_offsets(actual, trinity_dist)
    print(f"Trinity Average Calibrated Macro-F1: {trinity_score:.5f}")
    
    # Geometric Trinity
    geo_trinity = np.exp(
        (np.log(xgb_dist + 1e-7) + np.log(lgb_dist + 1e-7) + np.log(cb_dist + 1e-7)) / 3.0
    )
    geo_trinity /= geo_trinity.sum(axis=1, keepdims=True)
    geo_offsets, geo_score = tune_offsets(actual, geo_trinity)
    print(f"Geometric Trinity Calibrated Macro-F1: {geo_score:.5f}")
    
    preds = (np.log(np.clip(geo_trinity, 1e-7, 1.0)) + geo_offsets).argmax(axis=1)
    print("\nDetailed Per-Class Performance:")
    print(classification_report(actual, preds, target_names=LABELS, digits=5))

if __name__ == "__main__":
    main()
