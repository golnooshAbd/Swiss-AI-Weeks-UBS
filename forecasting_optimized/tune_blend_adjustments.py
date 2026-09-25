import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
import random

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables

def as_distribution(recurring: np.ndarray) -> np.ndarray:
    none = np.clip(1.0 - recurring.max(axis=1, keepdims=True), 1e-6, 1.0)
    probabilities = np.column_stack([recurring, none])
    return probabilities / probabilities.sum(axis=1, keepdims=True)

from typing import Optional
def macro_f1(actual: np.ndarray, scores: np.ndarray, offsets: Optional[np.ndarray] = None) -> float:
    adjusted = np.log(np.clip(scores, 1e-7, 1.0))
    if offsets is not None:
        adjusted = adjusted + offsets
    return float(
        f1_score(
            actual,
            adjusted.argmax(axis=1),
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )

def tune_offsets(actual: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    best_score = macro_f1(actual, probabilities, offsets)
    for step in (0.4, 0.2, 0.1, 0.05, 0.025):
        improved = True
        while improved:
            improved = False
            for column in range(len(LABELS)):
                for direction in (-step, step):
                    candidate = offsets.copy()
                    candidate[column] += direction
                    score = macro_f1(actual, probabilities, candidate)
                    if score > best_score + 1e-9:
                        offsets, best_score = candidate, score
                        improved = True
    return offsets, best_score

def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    artifact_dir = Path(__file__).parent / "artifacts"
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
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

    with np.load(artifact_dir / "family_valid_probabilities.npz") as family_data:
        family_by_client = {
            str(client_id): probabilities
            for client_id, probabilities in zip(
                family_data["client_ids"], family_data["probabilities"]
            )
        }
    family = as_distribution(np.stack([family_by_client[client_id] for client_id in wide.index]))

    with np.load(artifact_dir / "neural_valid_probabilities.npz") as neural_data:
        neural_by_client = {
            str(client_id): probabilities
            for client_id, probabilities in zip(
                neural_data["client_ids"], neural_data["probabilities"]
            )
        }
    neural = np.stack([neural_by_client[client_id] for client_id in wide.index])
    actual = target_by_client.loc[wide.index].to_numpy()

    print("Re-evaluating base blend...")
    
    weights = (0.3, 0.56, 0.0, 0.14)
    base_probabilities = (
        weights[0] * catboost
        + weights[1] * pair
        + weights[2] * family
        + weights[3] * neural
    )
    
    offsets, score = tune_offsets(actual, base_probabilities)
    print(f"Base score: {score:.4f}")

    extra_sources: dict[str, np.ndarray] = {}
    
    def load_npz(path, key="probabilities"):
        if not path.exists(): return None
        with np.load(path) as data:
            by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data[key])}
        return np.stack([by_client[cid] for cid in wide.index])

    extra_sources["stacking"] = load_npz(artifact_dir / "stacking_valid_probabilities.npz")
    
    proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(value) for value in data["client_ids"]]
            positions = {client_id: index for index, client_id in enumerate(proxy_order)}
            extra_sources["proxy_catboost"] = np.stack([data["catboost"][positions[cid]] for cid in wide.index])
            extra_sources["proxy_pair"] = np.stack([data["pair"][positions[cid]] for cid in wide.index])
            
    extra_sources["xgboost_pair"] = load_npz(artifact_dir / "xgboost_valid_probabilities.npz")
    extra_sources["ranking"] = load_npz(artifact_dir / "ranking_valid_probabilities.npz")
    extra_sources["embedding_pool"] = load_npz(artifact_dir / "embedding_pool_valid_probabilities.npz")
    extra_sources["description_stream"] = load_npz(artifact_dir / "description_stream_valid_probabilities.npz")
    extra_sources["catboost_tuned"] = load_npz(artifact_dir / "catboost_tuned_valid_probabilities.npz")
    
    extra_sources = {k: v for k, v in extra_sources.items() if v is not None}

    micro_probs = load_npz(artifact_dir / "micro_valid_probabilities.npz")

    print(f"Available extra sources: {list(extra_sources.keys())}")
    
    best_overall_score = score
    best_config = None
    
    active_sources = ["stacking", "xgboost_pair", "ranking", "embedding_pool", "description_stream"]
    active_sources = [s for s in active_sources if s in extra_sources]
    
    print("Running random search for extra weights and adjustments...")
    for iteration in range(5000):
        probabilities = base_probabilities.copy()
        current_extra_weights = {}
        
        for source_name in active_sources:
            if random.random() < 0.8:
                w = random.uniform(0.01, 0.4)
                current_extra_weights[source_name] = w
                probabilities = np.exp(
                    (1 - w) * np.log(probabilities + 1e-7)
                    + w * np.log(extra_sources[source_name] + 1e-7)
                )
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                
        micro_weight = 0.0
        if micro_probs is not None and random.random() < 0.7:
            micro_weight = random.uniform(0.05, 0.5)
            targets = [LABELS.index('music'), LABELS.index('streaming'), LABELS.index('software')]
            for t in targets:
                probabilities[:, t] = np.exp(
                    (1 - micro_weight) * np.log(probabilities[:, t] + 1e-7) + micro_weight * np.log(micro_probs[:, t] + 1e-7)
                )
            probabilities /= probabilities.sum(axis=1, keepdims=True)

        current_adjustments = []
        for _ in range(random.randint(2, 8)):
            source_name = random.choice(active_sources)
            label = random.choice(LABELS)
            column = LABELS.index(label)
            weight = random.uniform(-0.3, 0.5)
            
            probabilities[:, column] = (
                (1 - weight) * probabilities[:, column]
                + weight * extra_sources[source_name][:, column]
            )
            probabilities = np.clip(probabilities, 1e-7, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            
            current_adjustments.append({"source": source_name, "label": label, "weight": weight})
            
        candidate_offsets, candidate_score = tune_offsets(actual, probabilities)
        
        if candidate_score > best_overall_score:
            best_overall_score = candidate_score
            best_config = {
                "extra_weights": current_extra_weights,
                "micro_override_weight": micro_weight,
                "class_adjustments": current_adjustments,
                "class_offsets": dict(zip(LABELS, candidate_offsets.tolist()))
            }
            print(f"Iter {iteration}: New best score: {best_overall_score:.5f}")
            
    print(f"\nFinal Best Score: {best_overall_score:.5f}")
    if best_config:
        print("Config:")
        print(json.dumps(best_config, indent=2))
        
        with open("best_random_search_config.json", "w") as f:
            json.dump(best_config, f, indent=2)

if __name__ == "__main__":
    main()
