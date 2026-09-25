import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables

def as_distribution(recurring: np.ndarray) -> np.ndarray:
    none = np.clip(1.0 - recurring.max(axis=1, keepdims=True), 1e-6, 1.0)
    probabilities = np.column_stack([recurring, none])
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def load_npz(path, index_order, key="probabilities"):
    path = Path(path)
    if not path.exists(): return None
    with np.load(path) as data:
        by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data[key])}
    return np.stack([by_client[cid] for cid in index_order])

def main():
    repository = Path(__file__).resolve().parents[1]
    artifact_dir = Path(__file__).parent / "artifacts"
    feature_csv = repository / "data" / "dataset_features" / "valid_features.csv"
    labels_path = repository / "data" / "dataset" / "valid_labels.csv"

    labels = pd.read_csv(labels_path, dtype={"client_id": str})
    target_map = {name: index for index, name in enumerate(LABELS)}
    target = labels["target_next_recurring_merchant"].map(target_map)
    target_by_client = pd.Series(target.to_numpy(), index=labels["client_id"])

    # Base models
    bundle = joblib.load(artifact_dir / "model.joblib")
    frame = pd.read_csv(feature_csv)
    wide, pairs = build_feature_tables(frame, labels["client_id"])
    wide = wide.reindex(columns=bundle["wide_columns"]).fillna(0.0)
    pairs = pairs.reindex(columns=bundle["pair_columns"]).fillna(0.0)
    
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())
    
    with np.load(artifact_dir / "family_valid_probabilities.npz") as data:
        family_dict = {str(cid): p for cid, p in zip(data["client_ids"], data["probabilities"])}
    family = as_distribution(np.stack([family_dict[cid] for cid in wide.index]))
    
    with np.load(artifact_dir / "neural_valid_probabilities.npz") as data:
        neural_dict = {str(cid): p for cid, p in zip(data["client_ids"], data["probabilities"])}
    neural = np.stack([neural_dict[cid] for cid in wide.index])
    
    actual = target_by_client.loc[wide.index].to_numpy()
    
    # Load optimal config
    with open(artifact_dir / "blend_config.json") as f:
        config = json.load(f)
        
    w = config["weights"]
    probabilities = (
        w["catboost"] * catboost +
        w["pair"] * pair +
        w["family"] * family +
        w["neural"] * neural
    )
    
    # Extra models
    for name, weight in config.get("extra_weights", {}).items():
        if weight > 0:
            filename = "xgboost_valid_probabilities.npz" if name == "xgboost_pair" else f"{name}_valid_probabilities.npz"
            extra = load_npz(artifact_dir / filename, wide.index)
            if extra is None and name.startswith("proxy_"):
                proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
                if proxy_path.exists():
                    with np.load(proxy_path) as data:
                        proxy_order = [str(v) for v in data["client_ids"]]
                        positions = {cid: idx for idx, cid in enumerate(proxy_order)}
                        key = "catboost" if name == "proxy_catboost" else "pair"
                        extra = np.stack([data[key][positions[cid]] for cid in wide.index])
            if extra is not None:
                probabilities = np.exp(
                    (1 - weight) * np.log(probabilities + 1e-7) + 
                    weight * np.log(extra + 1e-7)
                )
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                
    # Micro override
    micro_weight = config.get("micro_override_weight", 0.0)
    if micro_weight > 0:
        micro = load_npz(artifact_dir / "micro_valid_probabilities.npz", wide.index)
        targets = [LABELS.index('music'), LABELS.index('streaming'), LABELS.index('software')]
        for t in targets:
            probabilities[:, t] = np.exp(
                (1 - micro_weight) * np.log(probabilities[:, t] + 1e-7) + 
                micro_weight * np.log(micro[:, t] + 1e-7)
            )
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
    # Adjustments
    for adj in config.get("class_adjustments", []):
        src = adj["source"]
        filename = "xgboost_valid_probabilities.npz" if src == "xgboost_pair" else f"{src}_valid_probabilities.npz"
        extra = load_npz(artifact_dir / filename, wide.index)
        if extra is None and src.startswith("proxy_"):
            proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
            if proxy_path.exists():
                with np.load(proxy_path) as data:
                    proxy_order = [str(v) for v in data["client_ids"]]
                    positions = {cid: idx for idx, cid in enumerate(proxy_order)}
                    key = "catboost" if src == "proxy_catboost" else "pair"
                    extra = np.stack([data[key][positions[cid]] for cid in wide.index])
        col = LABELS.index(adj["label"])
        probabilities[:, col] = (1 - adj["weight"]) * probabilities[:, col] + adj["weight"] * extra[:, col]
        probabilities = np.clip(probabilities, 1e-7, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
    # Offsets
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    for label, val in config.get("class_offsets", {}).items():
        offsets[LABELS.index(label)] = val
        
    adjusted = np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets
    predictions = adjusted.argmax(axis=1)
    
    print("\n" + "="*50)
    print("UBS Hackathon Evaluation - Final Blend vs Validation Set")
    print("="*50)
    print(classification_report(actual, predictions, target_names=LABELS, digits=5))
    print("="*50)

if __name__ == "__main__":
    main()
