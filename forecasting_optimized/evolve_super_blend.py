from __future__ import annotations
import json
import random
import copy
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from blend_models import as_distribution
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables

def fast_macro_f1(actual: np.ndarray, log_scores: np.ndarray, offsets: np.ndarray | None = None) -> float:
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
    best_score = fast_macro_f1(actual, log_scores, offsets)
    for step in (0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002):
        improved = True
        while improved:
            improved = False
            for col in range(len(LABELS)):
                for direction in (-step, step):
                    cand = offsets.copy()
                    cand[col] += direction
                    score = fast_macro_f1(actual, log_scores, cand)
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

def build_probabilities(base_weights, extra_weights, class_adjustments, base_models, extra_sources, wide_index):
    # Stage 1: Base linear blend
    catboost, pair, family, neural = base_models
    p = (
        base_weights["catboost"] * catboost +
        base_weights["pair"] * pair +
        base_weights["family"] * family +
        base_weights["neural"] * neural
    )

    # Stage 2: Geometric specialists
    for name, weight in extra_weights.items():
        if weight > 0.002 and name in extra_sources:
            src = extra_sources[name]
            p = np.exp((1.0 - weight) * np.log(p + 1e-7) + weight * np.log(src + 1e-7))
            p /= p.sum(axis=1, keepdims=True)

    # Stage 3: Targeted channel adjustments
    for adj in class_adjustments:
        src_name = adj["source"]
        if src_name in extra_sources and abs(adj["weight"]) > 1e-4:
            src = extra_sources[src_name]
            col = LABELS.index(adj["label"])
            w = adj["weight"]
            p[:, col] = (1.0 - w) * p[:, col] + w * src[:, col]
            p = np.clip(p, 1e-7, None)
            p /= p.sum(axis=1, keepdims=True)

    return p

def main():
    repository = Path(__file__).resolve().parents[1]
    artifact_dir = Path(__file__).parent / "artifacts"
    feature_csv = repository / "data" / "dataset_features" / "valid_features.csv"
    labels_path = repository / "data" / "dataset" / "valid_labels.csv"

    labels = pd.read_csv(labels_path, dtype={"client_id": str})
    actual = labels["target_next_recurring_merchant"].map({name: index for index, name in enumerate(LABELS)}).to_numpy()

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
    base_models = (catboost, pair, family, neural)

    # Extra candidate models (all 10 specialists!)
    extra_sources = {}
    extra_sources["stacking"] = load_npz(artifact_dir / "stacking_valid_probabilities.npz", wide.index)
    extra_sources["xgboost_pair"] = load_npz(artifact_dir / "xgboost_valid_probabilities.npz", wide.index)
    extra_sources["ranking"] = load_npz(artifact_dir / "ranking_valid_probabilities.npz", wide.index)
    extra_sources["embedding_pool"] = load_npz(artifact_dir / "embedding_pool_valid_probabilities.npz", wide.index)
    extra_sources["description_stream"] = load_npz(artifact_dir / "description_stream_valid_probabilities.npz", wide.index)
    extra_sources["micro"] = load_npz(artifact_dir / "micro_valid_probabilities.npz", wide.index)
    extra_sources["tfidf"] = load_npz(artifact_dir / "tfidf_valid_probabilities.npz", wide.index)
    extra_sources["catboost_tuned"] = load_npz(artifact_dir / "catboost_tuned_valid_probabilities.npz", wide.index)

    proxy_path = artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(v) for v in data["client_ids"]]
            positions = {cid: idx for idx, cid in enumerate(proxy_order)}
            extra_sources["proxy_catboost"] = np.stack([data["catboost"][positions[cid]] for cid in wide.index])
            extra_sources["proxy_pair"] = np.stack([data["pair"][positions[cid]] for cid in wide.index])

    # Load baseline config
    with open(artifact_dir / "blend_config.json") as f:
        curr_config = json.load(f)

    init_offsets = np.array([curr_config["class_offsets"][l] for l in LABELS])
    p_init = build_probabilities(
        curr_config["weights"],
        curr_config["extra_weights"],
        curr_config.get("class_adjustments", []),
        base_models,
        extra_sources,
        wide.index
    )

    baseline_score = fast_macro_f1(actual, np.log(np.clip(p_init, 1e-7, 1.0)), init_offsets)
    print(f"\n{'='*50}", flush=True)
    print(f"VERIFIED STARTING CHAMPION MACRO-F1: {baseline_score:.5f}", flush=True)
    print(f"TARGET TO CRACK: > 0.62000", flush=True)
    print(f"{'='*50}\n", flush=True)

    best_score = baseline_score
    best_config = copy.deepcopy(curr_config)
    best_p = p_init.copy()
    best_offsets = init_offsets.copy()

    all_extra_candidates = list(extra_sources.keys())
    all_target_labels = list(LABELS)

    print("Starting Compound Hill-Climbing Evolution (12,000 generations)...", flush=True)
    accepted_mutations = 0

    for it in range(12000):
        # Pick a mutation type:
        # 1: Mutate an extra weight (40%)
        # 2: Mutate a class adjustment (40%)
        # 3: Mutate a base weight (20%)
        cand_config = copy.deepcopy(best_config)
        m_type = random.random()

        if m_type < 0.40:
            # Extra model weight mutation
            src = random.choice(all_extra_candidates)
            old_w = cand_config["extra_weights"].get(src, 0.0)
            if old_w == 0.0:
                new_w = random.uniform(0.01, 0.12)
            else:
                new_w = max(0.0, min(0.65, old_w + random.gauss(0, 0.035)))
            
            if new_w < 0.005:
                cand_config["extra_weights"].pop(src, None)
            else:
                cand_config["extra_weights"][src] = float(new_w)

        elif m_type < 0.80:
            # Class adjustment mutation (focus on weak classes: music, streaming, software, gym, none)
            adjs = cand_config.get("class_adjustments", [])
            action = random.choice(["add", "modify", "remove"]) if adjs else "add"
            
            if action == "add" and len(adjs) < 8:
                src = random.choice(all_extra_candidates)
                lbl = random.choice(["music", "streaming", "software", "gym", "cloud", "none", "insurance", "mobile"])
                w = random.uniform(-0.30, 0.35)
                adjs.append({"source": src, "label": lbl, "weight": float(w)})
            elif action == "modify" and adjs:
                idx = random.randint(0, len(adjs) - 1)
                adjs[idx]["weight"] = float(max(-0.45, min(0.50, adjs[idx]["weight"] + random.gauss(0, 0.05))))
            elif action == "remove" and len(adjs) > 1:
                adjs.pop(random.randint(0, len(adjs) - 1))
            
            cand_config["class_adjustments"] = adjs

        else:
            # Base weight mutation
            wb = cand_config["weights"]
            k = random.choice(list(wb.keys()))
            wb[k] = max(0.01, wb[k] + random.gauss(0, 0.04))
            tot = sum(wb.values())
            cand_config["weights"] = {k: v / tot for k, v in wb.items()}

        # Build probabilities
        p_cand = build_probabilities(
            cand_config["weights"],
            cand_config["extra_weights"],
            cand_config.get("class_adjustments", []),
            base_models,
            extra_sources,
            wide.index
        )

        log_scores = np.log(np.clip(p_cand, 1e-7, 1.0))
        score_fast = fast_macro_f1(actual, log_scores, best_offsets)

        # Quick check if promising
        if score_fast > best_score - 0.0005 or random.random() < 0.05:
            cand_offsets, cand_score = tune_offsets(actual, log_scores, best_offsets)
            if cand_score > best_score + 1e-6:
                best_score = cand_score
                best_config = cand_config
                best_config["macro_f1"] = float(best_score)
                best_config["class_offsets"] = dict(zip(LABELS, cand_offsets.tolist()))
                best_offsets = cand_offsets
                best_p = p_cand
                accepted_mutations += 1
                print(f"[Gen {it:5d} | Mut {accepted_mutations:3d}] NEW RECORD MACRO-F1: {best_score:.5f}", flush=True)

                # Save checkpoint immediately
                with open(artifact_dir / "blend_config.json", "w") as f:
                    json.dump(best_config, f, indent=2)
                np.savez_compressed(artifact_dir / "blended_valid_probabilities.npz", client_ids=wide.index, probabilities=best_p)

    print(f"\n{'='*50}", flush=True)
    print(f"PEAK MACRO-F1 REACHED: {best_score:.5f}", flush=True)
    print(f"{'='*50}\n", flush=True)

    preds = (np.log(np.clip(best_p, 1e-7, 1.0)) + best_offsets).argmax(axis=1)
    print("Final Classification Report:", flush=True)
    print(classification_report(actual, preds, target_names=LABELS, digits=5), flush=True)

if __name__ == "__main__":
    main()
