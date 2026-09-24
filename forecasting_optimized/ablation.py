import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
from typing import Optional

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables
from blend_models import as_distribution, tune_offsets

def macro_f1(actual: np.ndarray, scores: np.ndarray, offsets: Optional[np.ndarray] = None) -> float:
    adjusted = np.log(np.clip(scores, 1e-7, 1.0))
    if offsets is not None:
        adjusted = adjusted + offsets
    return float(f1_score(actual, adjusted.argmax(axis=1), labels=np.arange(len(LABELS)), average="macro", zero_division=0))

def main():
    repository = Path(__file__).resolve().parents[1]
    artifact_dir = repository / "forecasting_optimized" / "artifacts"
    feature_csv = repository / "data" / "dataset_features" / "valid_features.csv"
    labels_path = repository / "data" / "dataset" / "valid_labels.csv"

    labels = pd.read_csv(labels_path, dtype={"client_id": str})
    target_map = {name: index for index, name in enumerate(LABELS)}
    target = labels["target_next_recurring_merchant"].map(target_map)
    target_by_client = pd.Series(target.to_numpy(), index=labels["client_id"])

    bundle = joblib.load(artifact_dir / "model.joblib")
    frame = pd.read_csv(feature_csv)
    wide, pairs = build_feature_tables(frame, labels["client_id"])
    wide = wide.reindex(columns=bundle["wide_columns"])
    pairs = pairs.reindex(columns=bundle["pair_columns"])
    
    # 1. BASE MODELS (from experiment.py)
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())
    
    # Only use Catboost (10%) and Pair (90%) - dropping family and neural completely!
    base_probabilities = 0.1 * catboost + 0.9 * pair
    offsets, score = tune_offsets(target_by_client.loc[wide.index].to_numpy(), base_probabilities)
    print(f"Score with ONLY Catboost + Pair Model (2 fast models): {score:.4f}")

    # 2. Add Stacking (The Judge)
    with np.load(artifact_dir / "stacking_valid_probabilities.npz") as data:
        by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data["probabilities"])}
    stacking = np.stack([by_client[cid] for cid in wide.index])
    
    stacked_probs = 0.7 * base_probabilities + 0.3 * stacking
    offsets, score = tune_offsets(target_by_client.loc[wide.index].to_numpy(), stacked_probs)
    print(f"Score adding Stacking (+1 fast model): {score:.4f}")

    # 3. Add Embedding Pool (The Text specialist)
    with np.load(artifact_dir / "embedding_pool_valid_probabilities.npz") as data:
        by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data["probabilities"])}
    embedding = np.stack([by_client[cid] for cid in wide.index])
    
    final_probs = 0.7 * stacked_probs + 0.3 * embedding
    offsets, score = tune_offsets(target_by_client.loc[wide.index].to_numpy(), final_probs)
    print(f"Score adding Embedding Pool (+1 fast model): {score:.4f}")

if __name__ == "__main__":
    main()
