import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
import optuna

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables

def as_distribution(recurring: np.ndarray) -> np.ndarray:
    none = np.clip(1.0 - recurring.max(axis=1, keepdims=True), 1e-6, 1.0)
    probabilities = np.column_stack([recurring, none])
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def macro_f1(actual: np.ndarray, scores: np.ndarray, offsets: np.ndarray) -> float:
    adjusted = np.log(np.clip(scores, 1e-7, 1.0)) + offsets
    return float(
        f1_score(
            actual,
            adjusted.argmax(axis=1),
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )

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
    
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

    sources = {"catboost": catboost, "pair": pair}

    def load_npz(path, key="probabilities"):
        if not path.exists(): return None
        with np.load(path) as data:
            by_client = {str(cid): vals for cid, vals in zip(data["client_ids"], data[key])}
        return np.stack([by_client[cid] for cid in wide.index])

    sources["family"] = as_distribution(load_npz(artifact_dir / "family_valid_probabilities.npz"))
    sources["neural"] = load_npz(artifact_dir / "neural_valid_probabilities.npz")
    sources["stacking"] = load_npz(artifact_dir / "stacking_valid_probabilities.npz")
    
    proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(value) for value in data["client_ids"]]
            positions = {client_id: index for index, client_id in enumerate(proxy_order)}
            sources["proxy_catboost"] = np.stack([data["catboost"][positions[cid]] for cid in wide.index])
            sources["proxy_pair"] = np.stack([data["pair"][positions[cid]] for cid in wide.index])
            
    sources["xgboost_pair"] = load_npz(artifact_dir / "xgboost_valid_probabilities.npz")
    sources["ranking"] = load_npz(artifact_dir / "ranking_valid_probabilities.npz")
    sources["embedding_pool"] = load_npz(artifact_dir / "embedding_pool_valid_probabilities.npz")
    sources["description_stream"] = load_npz(artifact_dir / "description_stream_valid_probabilities.npz")

    actual = target_by_client.loc[wide.index].to_numpy()

    def objective(trial):
        probs = np.zeros_like(catboost)
        total_weight = 0
        for name, p in sources.items():
            if p is not None:
                w = trial.suggest_float(f"weight_{name}", 0.0, 1.0)
                probs += w * p
                total_weight += w
        probs /= total_weight
        
        offsets = np.zeros(len(LABELS))
        for i, label in enumerate(LABELS):
            offsets[i] = trial.suggest_float(f"offset_{label}", -1.0, 1.0)
            
        return macro_f1(actual, probs, offsets)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5000, n_jobs=-1)
    
    print(f"Best Optuna Macro-F1: {study.best_value:.4f}")
    
if __name__ == "__main__":
    main()
