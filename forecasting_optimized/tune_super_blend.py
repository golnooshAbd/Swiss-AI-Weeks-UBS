from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from blend_models import as_distribution
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables

def macro_f1(actual: np.ndarray, log_scores: np.ndarray, offsets: np.ndarray | None = None) -> float:
    scores = log_scores + offsets if offsets is not None else log_scores
    preds = scores.argmax(axis=1)
    f1s = []
    for c in range(8):
        tp = np.sum((preds == c) & (actual == c))
        fp = np.sum((preds == c) & (actual != c))
        fn = np.sum((preds != c) & (actual == c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))

def tune_offsets(actual: np.ndarray, log_scores: np.ndarray, init_offsets: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    offsets = np.zeros(len(LABELS), dtype=np.float64) if init_offsets is None else init_offsets.copy()
    best_score = macro_f1(actual, log_scores, offsets)
    for step in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for col in range(len(LABELS)):
                for direction in (-step, step):
                    cand = offsets.copy()
                    cand[col] += direction
                    score = macro_f1(actual, log_scores, cand)
                    if score > best_score + 1e-7:
                        offsets, best_score = cand, score
                        improved = True
    return offsets, best_score

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
    actual = labels["target_next_recurring_merchant"].map(target_map).to_numpy()

    bundle = joblib.load(artifact_dir / "model.joblib")
    frame = pd.read_csv(feature_csv)
    wide, pairs = build_feature_tables(frame, labels["client_id"])
    wide = wide.reindex(columns=bundle["wide_columns"]).fillna(0.0)
    pairs = pairs.reindex(columns=bundle["pair_columns"]).fillna(0.0)

    # Base models
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())
    family = as_distribution(load_npz(artifact_dir / "family_valid_probabilities.npz", wide.index))
    neural = load_npz(artifact_dir / "neural_valid_probabilities.npz", wide.index)

    # Extra candidate models
    extra_sources = {}
    extra_sources["stacking"] = load_npz(artifact_dir / "stacking_valid_probabilities.npz", wide.index)
    extra_sources["xgboost_pair"] = load_npz(artifact_dir / "xgboost_valid_probabilities.npz", wide.index)
    extra_sources["ranking"] = load_npz(artifact_dir / "ranking_valid_probabilities.npz", wide.index)
    extra_sources["embedding_pool"] = load_npz(artifact_dir / "embedding_pool_valid_probabilities.npz", wide.index)
    extra_sources["description_stream"] = load_npz(artifact_dir / "description_stream_valid_probabilities.npz", wide.index)
    extra_sources["micro"] = load_npz(artifact_dir / "micro_valid_probabilities.npz", wide.index)
    extra_sources["tfidf"] = load_npz(artifact_dir / "tfidf_valid_probabilities.npz", wide.index)
    extra_sources["catboost_tuned"] = load_npz(artifact_dir / "catboost_tuned_valid_probabilities.npz", wide.index)

    # Proxy models
    proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(v) for v in data["client_ids"]]
            positions = {cid: idx for idx, cid in enumerate(proxy_order)}
            extra_sources["proxy_catboost"] = np.stack([data["catboost"][positions[cid]] for cid in wide.index])
            extra_sources["proxy_pair"] = np.stack([data["pair"][positions[cid]] for cid in wide.index])

    # Load baseline config
    with open(artifact_dir / "blend_config.json") as f:
        base_config = json.load(f)

    print("Evaluating current baseline configuration...")
    # Base linear
    w = base_config["weights"]
    p_base = w["catboost"] * catboost + w["pair"] * pair + w["family"] * family + w["neural"] * neural
    p = p_base.copy()

    for name, weight in base_config["extra_weights"].items():
        src = extra_sources.get(name)
        if src is not None and weight > 0:
            p = np.exp((1 - weight) * np.log(p + 1e-7) + weight * np.log(src + 1e-7))
            p /= p.sum(axis=1, keepdims=True)

    for adj in base_config.get("class_adjustments", []):
        src = extra_sources.get(adj["source"])
        if src is not None:
            col = LABELS.index(adj["label"])
            p[:, col] = (1 - adj["weight"]) * p[:, col] + adj["weight"] * src[:, col]
            p = np.clip(p, 1e-7, None)
            p /= p.sum(axis=1, keepdims=True)

    init_offsets = np.array([base_config["class_offsets"][l] for l in LABELS])
    baseline_score = macro_f1(actual, np.log(np.clip(p, 1e-7, 1.0)), init_offsets)
    print(f"Current Verified Baseline Macro-F1: {baseline_score:.5f}")

    # Now let's optimize around the baseline with targeted search
    import random
    best_score = baseline_score
    best_config = base_config.copy()
    best_probs = p.copy()
    best_offsets = init_offsets.copy()

    print("\nStarting Targeted Search to Beat 0.60847...")

    candidate_sources = list(extra_sources.keys())
    extra_model_list = [
        "stacking", "xgboost_pair", "ranking", "embedding_pool",
        "description_stream", "micro", "tfidf", "catboost_tuned",
        "proxy_catboost", "proxy_pair"
    ]
    
    for it in range(25000):
        # Mutate base weights slightly around (0.30, 0.56, 0.0, 0.14)
        wb = {
            "catboost": max(0.05, 0.30 + random.uniform(-0.12, 0.12)),
            "pair": max(0.20, 0.56 + random.uniform(-0.14, 0.14)),
            "family": max(0.0, random.uniform(0.0, 0.10)),
            "neural": max(0.05, 0.14 + random.uniform(-0.08, 0.08))
        }
        total_wb = sum(wb.values())
        wb = {k: v / total_wb for k, v in wb.items()}

        p_curr = wb["catboost"] * catboost + wb["pair"] * pair + wb["family"] * family + wb["neural"] * neural

        # Extra weights mutation around winning values
        curr_extra_w = {}
        for s in extra_model_list:
            old_w = base_config["extra_weights"].get(s, 0.0)
            if s in ["proxy_catboost", "proxy_pair", "micro", "tfidf", "catboost_tuned"] and old_w == 0.0:
                new_w = random.uniform(0.0, 0.18) if random.random() < 0.4 else 0.0
            else:
                new_w = max(0.0, min(0.65, old_w + random.uniform(-0.08, 0.08)))
            
            if new_w > 0.005 and s in extra_sources:
                curr_extra_w[s] = new_w
                src = extra_sources[s]
                p_curr = np.exp((1 - new_w) * np.log(p_curr + 1e-7) + new_w * np.log(src + 1e-7))
                p_curr /= p_curr.sum(axis=1, keepdims=True)

        # Targeted class adjustments (especially for weak classes: music, streaming, software, gym, none)
        curr_adj = []
        n_adj = random.randint(1, 5)
        for _ in range(n_adj):
            src_name = random.choice(candidate_sources)
            tgt_label = random.choice(["music", "streaming", "software", "gym", "cloud", "none", "insurance", "mobile"])
            tgt_col = LABELS.index(tgt_label)
            w_adj = random.uniform(-0.35, 0.40)
            curr_adj.append({"source": src_name, "label": tgt_label, "weight": w_adj})
            p_curr[:, tgt_col] = (1 - w_adj) * p_curr[:, tgt_col] + w_adj * extra_sources[src_name][:, tgt_col]
            p_curr = np.clip(p_curr, 1e-7, None)
            p_curr /= p_curr.sum(axis=1, keepdims=True)

        log_scores = np.log(np.clip(p_curr, 1e-7, 1.0))
        cand_offsets, cand_score = tune_offsets(actual, log_scores, init_offsets)

        if cand_score > best_score + 1e-6:
            best_score = cand_score
            best_offsets = cand_offsets
            best_probs = p_curr
            best_config = {
                "macro_f1": float(best_score),
                "weights": wb,
                "extra_weights": curr_extra_w,
                "class_adjustments": curr_adj,
                "class_offsets": dict(zip(LABELS, cand_offsets.tolist()))
            }
            print(f"Iter {it:5d}: NEW RECORD MACRO-F1: {best_score:.5f}", flush=True)

    print(f"\n==================================================")
    print(f"PEAK MACRO-F1 REACHED: {best_score:.5f}")
    print(f"==================================================")

    if best_score > baseline_score:
        preds = (np.log(np.clip(best_probs, 1e-7, 1.0)) + best_offsets).argmax(axis=1)
        print("\nNew Per-Class Classification Report:")
        print(classification_report(actual, preds, target_names=LABELS, digits=5))

        with open(artifact_dir / "blend_config.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print("Updated artifacts/blend_config.json with the new winning configuration!")

        # Save predictions
        np.savez_compressed(artifact_dir / "blended_valid_probabilities.npz", client_ids=wide.index, probabilities=best_probs)

if __name__ == "__main__":
    main()
