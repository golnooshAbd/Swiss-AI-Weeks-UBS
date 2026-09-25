"""
Retrain the full UBS transaction forecasting ensemble on 100% of labeled data
(Train [1] + Validation [2] = 3,000 clients) and generate the final competition
predictions on Test [3].
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier, XGBRanker

# Local imports from forecasting_optimized
from blend_models import as_distribution
from description_stream_experiment import description_features
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables
from neural_model import TemporalTransformerForecastModel
from stacking_experiment import meta_features
from txembed import (
    PreprocessedTransactions,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)
from txembed.batching import TransactionSequence


class LabeledSequences(Dataset):
    def __init__(self, sequences: TransactionSequenceDataset, labels: dict[str, int]) -> None:
        self.sequences = sequences
        self.labels = [labels[client_id] for client_id in sequences.client_ids]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[TransactionSequence, int]:
        return self.sequences[index], self.labels[index]


def collate_fn(items):
    sequences, labels = zip(*items)
    return (
        collate_transaction_sequences(list(sequences)),
        torch.tensor(labels, dtype=torch.long),
    )


def aggregate_text(frame: pd.DataFrame) -> pd.Series:
    data = frame[["client_id", "clean_description"]].copy()
    data["clean_description"] = data["clean_description"].fillna("").astype(str)
    return data.groupby("client_id")["clean_description"].apply(lambda texts: " ".join(texts))


def main():
    repo = Path(__file__).resolve().parents[1]
    feature_dir = repo / "data" / "dataset_features"
    label_dir = repo / "data" / "dataset"
    txembed_dir = repo / "transaction_embedding" / "artifacts" / "uncleaned"
    orig_artifact_dir = Path(__file__).parent / "artifacts"
    out_dir = Path(__file__).parent / "artifacts_retrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING FULL RE-TRAINING ON TRAIN [1] + VALIDATION [2] (3,000 CLIENTS)")
    print("=" * 60)

    # 1. Load and combine data
    print("\n--- Phase 1: Merging Datasets ---")
    t_feat = pd.read_csv(feature_dir / "train_features.csv")
    v_feat = pd.read_csv(feature_dir / "valid_features.csv")
    full_features = pd.concat([t_feat, v_feat], ignore_index=True)

    t_lab = pd.read_csv(label_dir / "train_labels.csv", dtype={"client_id": str})
    v_lab = pd.read_csv(label_dir / "valid_labels.csv", dtype={"client_id": str})
    full_labels = pd.concat([t_lab, v_lab], ignore_index=True)
    full_clients = full_labels["client_id"]
    full_targets = full_labels.set_index("client_id")["target_next_recurring_merchant"]
    full_target_names = full_labels["target_next_recurring_merchant"].to_numpy()

    label_map = {name: i for i, name in enumerate(LABELS)}
    y_full = full_targets.map(label_map).to_numpy()

    print(f"Combined features rows: {len(full_features):,}")
    print(f"Combined clients count: {len(full_clients):,}")

    # Copy static configurations
    shutil.copy(orig_artifact_dir / "blend_config.json", out_dir / "blend_config.json")
    if (orig_artifact_dir / "text_svd_pipeline.joblib").exists():
        shutil.copy(orig_artifact_dir / "text_svd_pipeline.joblib", out_dir / "text_svd_pipeline.joblib")
    if (orig_artifact_dir / "proxy_pretrained_models.joblib").exists():
        shutil.copy(orig_artifact_dir / "proxy_pretrained_models.joblib", out_dir / "proxy_pretrained_models.joblib")
    if (orig_artifact_dir / "embedding_pool_test_features.npz").exists():
        shutil.copy(orig_artifact_dir / "embedding_pool_test_features.npz", out_dir / "embedding_pool_test_features.npz")

    # 2. Build feature tables
    print("\n--- Phase 2: Building Tabular Feature Tables (Wide & Pairs) ---")
    orig_bundle = joblib.load(orig_artifact_dir / "model.joblib")
    wide, pairs = build_feature_tables(full_features, full_clients)
    wide = wide.reindex(columns=orig_bundle["wide_columns"]).fillna(0.0)
    pairs = pairs.reindex(columns=orig_bundle["pair_columns"]).fillna(0.0)
    print(f"Wide table shape: {wide.shape}, Pairs table shape: {pairs.shape}")

    pair_targets = np.asarray(
        [full_targets[cid] == fam for cid, fam in pairs.index],
        dtype=np.int8,
    )
    pair_weights = compute_sample_weight("balanced", pair_targets)
    wide_weights = compute_sample_weight("balanced", y_full)

    # 3. Model 1: Base Bundle (CatBoost Multiclass on Wide + HistGradientBoosting on Pairs)
    print("\n--- Phase 3: Base Model Bundle (CatBoost + Pair HGB) ---")
    if (out_dir / "model.joblib").exists():
        print("  model.joblib already cached.")
    else:
        cat_multi = CatBoostClassifier(
            iterations=1300,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=2026,
            verbose=False,
            thread_count=-1,
        )
        cat_multi.fit(wide, full_target_names, sample_weight=wide_weights, verbose=False)
        print("  -> CatBoost Multiclass trained with string class labels.")

        pair_hgb = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=450,
            max_leaf_nodes=23,
            min_samples_leaf=18,
            l2_regularization=1.0,
            random_state=2026,
        )
        pair_hgb.fit(pairs, pair_targets, sample_weight=pair_weights)
        print("  -> Pair HistGradientBoosting trained.")

        joblib.dump(
            {
                "catboost_multiclass_model": cat_multi,
                "pair_model": pair_hgb,
                "wide_columns": wide.columns.tolist(),
                "pair_columns": pairs.columns.tolist(),
            },
            out_dir / "model.joblib",
        )
        print("  Saved model.joblib.")

    # 4. Model 2: Family Bundle (7 CatBoost Binary Classifiers)
    print("\n--- Phase 4: Family-Specific Recurrence Models (7 Models) ---")
    if (out_dir / "family_models.joblib").exists():
        print("  family_models.joblib already cached.")
    else:
        family_models = {}
        family_columns = None
        for family in FAMILIES:
            fam_pairs = pairs.xs(family, level="family").reindex(full_clients)
            fam_binary = (full_targets.reindex(full_clients) == family).astype(int)
            family_columns = fam_pairs.columns.tolist()
            fam_model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=1000,
                learning_rate=0.035,
                depth=6,
                l2_leaf_reg=6.0,
                random_strength=0.7,
                auto_class_weights="Balanced",
                random_seed=2026,
                allow_writing_files=False,
                verbose=False,
                thread_count=-1,
            )
            fam_model.fit(fam_pairs, fam_binary, verbose=False)
            family_models[family] = fam_model
            print(f"  -> Family model '{family}' trained.")

        joblib.dump(
            {"models": family_models, "columns": family_columns},
            out_dir / "family_models.joblib",
        )
        print("  Saved family_models.joblib.")

    # 5. Model 3: XGBoost Pair Model
    print("\n--- Phase 5: XGBoost Pair Model ---")
    if (out_dir / "xgboost_pair_model.joblib").exists():
        print("  xgboost_pair_model.joblib already cached.")
    else:
        orig_xgb = joblib.load(orig_artifact_dir / "xgboost_pair_model.joblib")
        xgb_columns = orig_xgb["columns"]
        xgb_pairs = pairs.reindex(columns=xgb_columns).fillna(0.0)
        xgb_pair_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1800,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=3.0,
            gamma=0.02,
            tree_method="hist",
            random_state=2026,
            n_jobs=-1,
        )
        xgb_pair_model.fit(xgb_pairs, pair_targets, sample_weight=pair_weights, verbose=False)
        joblib.dump(
            {"model": xgb_pair_model, "columns": xgb_columns, "offsets": orig_xgb.get("offsets")},
            out_dir / "xgboost_pair_model.joblib",
        )
        print("  Saved xgboost_pair_model.joblib.")

    # 6. Model 4: Ranking Bundle (XGBRanker + XGB None Model)
    print("\n--- Phase 6: Ranking Model Bundle ---")
    if (out_dir / "ranking_model.joblib").exists():
        print("  ranking_model.joblib already cached.")
    else:
        orig_ranking = joblib.load(orig_artifact_dir / "ranking_model.joblib")
        recurring_clients = full_targets[full_targets != "none"].index
        ranking_train = pairs[pairs.index.get_level_values("client_id").isin(recurring_clients)]
        ranking_target = np.asarray(
            [full_targets[cid] == fam for cid, fam in ranking_train.index],
            dtype=np.int8,
        )
        ranker = XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg@1",
            n_estimators=1400,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=2026,
            n_jobs=-1,
        )
        ranker.fit(
            ranking_train,
            ranking_target,
            group=np.full(len(recurring_clients), len(FAMILIES), dtype=np.int32),
            verbose=False,
        )
        none_target = (full_targets.loc[wide.index] == "none").astype(np.int8)
        none_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1600,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=10.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=2026,
            n_jobs=-1,
        )
        none_model.fit(wide, none_target, verbose=False)
        joblib.dump(
            {
                "ranker": ranker,
                "none_model": none_model,
                "temperature": orig_ranking.get("temperature", 1.25),
                "offsets": orig_ranking.get("offsets"),
                "wide_columns": orig_ranking["wide_columns"],
                "pair_columns": orig_ranking["pair_columns"],
            },
            out_dir / "ranking_model.joblib",
        )
        print("  Saved ranking_model.joblib.")

    # 7. Model 5: Description Stream Model
    print("\n--- Phase 7: Description Stream Model ---")
    if (out_dir / "description_stream_model.joblib").exists():
        print("  description_stream_model.joblib already cached.")
    else:
        orig_desc = joblib.load(orig_artifact_dir / "description_stream_model.joblib")
        desc_cols = orig_desc["columns"]
        full_desc_features = description_features(full_features, full_clients)
        desc_pairs = pairs.join(full_desc_features).reindex(columns=desc_cols).fillna(0.0)
        desc_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=2026,
            n_jobs=-1,
        )
        desc_model.fit(desc_pairs, pair_targets, sample_weight=pair_weights, verbose=False)
        joblib.dump(
            {"model": desc_model, "columns": desc_cols, "offsets": orig_desc.get("offsets")},
            out_dir / "description_stream_model.joblib",
        )
        print("  Saved description_stream_model.joblib.")

    # 8. Model 6: Embedding Pool Model
    print("\n--- Phase 8: Embedding Pool Model ---")
    if (out_dir / "embedding_pool_model.joblib").exists():
        print("  embedding_pool_model.joblib already cached.")
    else:
        orig_pool_data = np.load(orig_artifact_dir / "embedding_pool_features.npz")
        pool_features = np.concatenate([orig_pool_data["train_features"], orig_pool_data["valid_features"]])
        pool_clients = np.concatenate([orig_pool_data["train_clients"], orig_pool_data["valid_clients"]]).astype(str)
        pool_by_client = {cid: feat for cid, feat in zip(pool_clients, pool_features)}
        X_pool = np.stack([pool_by_client[cid] for cid in full_clients]).astype(np.float32)

        pool_model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=2026,
            n_jobs=-1,
        )
        pool_model.fit(X_pool, y_full, sample_weight=wide_weights, verbose=False)
        joblib.dump({"model": pool_model}, out_dir / "embedding_pool_model.joblib")
        print("  Saved embedding_pool_model.joblib.")

    # 9. Model 7: TF-IDF Text Model
    print("\n--- Phase 9: TF-IDF Text Model ---")
    full_texts = aggregate_text(full_features).reindex(full_clients, fill_value="")
    if (out_dir / "tfidf_model.joblib").exists():
        print("  tfidf_model.joblib already cached.")
    else:
        tfidf_pipe = make_pipeline(
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                sublinear_tf=True,
            ),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=2026,
                n_jobs=-1,
            ),
        )
        tfidf_pipe.fit(full_texts, y_full)
        joblib.dump({"model": tfidf_pipe}, out_dir / "tfidf_model.joblib")
        print("  Saved tfidf_model.joblib.")

    # 10. Model 8: Micro-Classifier (music, streaming, software)
    print("\n--- Phase 10: Micro-Classifier (music/streaming/software) ---")
    if (out_dir / "micro_model.joblib").exists():
        print("  micro_model.joblib already cached.")
    else:
        micro_mask = full_targets.isin(["music", "streaming", "software"]).to_numpy()
        micro_clients = full_clients.to_numpy()[micro_mask]
        micro_y = y_full[micro_mask]
        micro_texts = full_texts.reindex(micro_clients, fill_value="")

        micro_pipe = make_pipeline(
            FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.9, sublinear_tf=True)),
            ]),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=2.0,
                random_state=2026,
                n_jobs=-1,
            ),
        )
        micro_pipe.fit(micro_texts, micro_y)
        joblib.dump({"model": micro_pipe}, out_dir / "micro_model.joblib")
        print("  Saved micro_model.joblib.")

    # 11. Model 9: Stacking Meta-Learner (5-fold Out-Of-Fold on full 3,000 clients)
    print("\n--- Phase 11: Training Stacking Ensemble Meta-Learner (5-Fold CV) ---")
    if (out_dir / "stacking_model.joblib").exists():
        print("  stacking_model.joblib already cached.")
    else:
        orig_stack = joblib.load(orig_artifact_dir / "stacking_model.joblib")
        stack_wide = wide.reindex(columns=orig_stack["wide_columns"]).fillna(0.0)
        stack_pairs = pairs.reindex(columns=orig_stack["pair_columns"]).fillna(0.0)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

        pair_models = []
        catboost_models = []
        oof_pair = np.zeros((len(full_clients), len(FAMILIES)), dtype=np.float64)
        oof_cat = np.zeros((len(full_clients), len(LABELS)), dtype=np.float64)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(stack_wide, y_full), start=1):
            trn_clients = full_clients.iloc[trn_idx]
            val_clients = full_clients.iloc[val_idx]

            # Fold pair model
            trn_pairs_fold = stack_pairs.loc[trn_clients]
            val_pairs_fold = stack_pairs.loc[val_clients]
            trn_bin = np.asarray([full_targets[cid] == fam for cid, fam in trn_pairs_fold.index], dtype=np.int8)
            w_bin = compute_sample_weight("balanced", trn_bin)

            pm = HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=300,
                max_leaf_nodes=23,
                min_samples_leaf=18,
                l2_regularization=1.0,
                random_state=2026 + fold,
            )
            pm.fit(trn_pairs_fold, trn_bin, sample_weight=w_bin)
            oof_pair[val_idx] = pair_probabilities(pm, val_pairs_fold).reindex(val_clients).to_numpy()
            pair_models.append(pm)

            # Fold CatBoost model - fit with string class names
            cm = CatBoostClassifier(
                iterations=900,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=5.0,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=2026 + fold,
                allow_writing_files=False,
                verbose=False,
                thread_count=-1,
            )
            cm.fit(stack_wide.iloc[trn_idx], full_target_names[trn_idx], verbose=False)
            oof_cat[val_idx] = probabilities_in_label_order(cm, stack_wide.iloc[val_idx])
            catboost_models.append(cm)
            print(f"  -> Fold {fold}/5 complete.")

        meta_X = meta_features(oof_pair, oof_cat)
        meta_model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=2026,
        )
        meta_model.fit(meta_X, y_full)
        joblib.dump(
            {
                "pair_models": pair_models,
                "catboost_models": catboost_models,
                "meta_model": meta_model,
                "meta_name": "logistic_0.5",
                "wide_columns": orig_stack["wide_columns"],
                "pair_columns": orig_stack["pair_columns"],
            },
            out_dir / "stacking_model.joblib",
        )
        print("  Saved stacking_model.joblib.")

    # 12. Model 10: PyTorch Temporal Transformer
    print("\n--- Phase 12: Training Temporal Transformer on 3,000 Sequences ---")
    if (out_dir / "neural_model.pt").exists():
        print("  neural_model.pt already cached.")
    else:
        train_rows = PreprocessedTransactions.load_npz(txembed_dir / "train.npz")
        valid_rows = PreprocessedTransactions.load_npz(txembed_dir / "valid.npz")
        combined_rows = PreprocessedTransactions(
            description_embeddings=np.concatenate([train_rows.description_embeddings, valid_rows.description_embeddings]),
            categorical_indices=np.concatenate([train_rows.categorical_indices, valid_rows.categorical_indices]),
            dense_features=np.concatenate([train_rows.dense_features, valid_rows.dense_features]),
            client_ids=np.concatenate([train_rows.client_ids, valid_rows.client_ids]),
            timestamps_ns=np.concatenate([train_rows.timestamps_ns, valid_rows.timestamps_ns]),
        )
        preprocessor = TransactionPreprocessor.load(txembed_dir / "preprocessor.json")
        sequences = TransactionSequenceDataset(combined_rows, max_length=128)
        label_dict = {cid: label_map[tgt] for cid, tgt in full_targets.items()}
        dataset = LabeledSequences(sequences, label_dict)
        loader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_model = TemporalTransformerForecastModel.from_preprocessor(preprocessor).to(device)
        counts = np.bincount(y_full, minlength=len(LABELS)).astype(np.float64)
        weights = np.sqrt(len(y_full) / (len(LABELS) * counts))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(neural_model.parameters(), lr=7e-4, weight_decay=2e-4)

        neural_model.train()
        print("  Starting 20 epochs of neural sequence training...")
        for epoch in range(1, 21):
            total_loss = 0.0
            for batch_seqs, batch_labels in loader:
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = neural_model(batch_seqs)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_labels)
            avg_loss = total_loss / len(dataset)
            if epoch % 5 == 0 or epoch == 20:
                print(f"    Epoch {epoch:2d}/20 - CrossEntropyLoss: {avg_loss:.4f}")

        torch.save(neural_model.state_dict(), out_dir / "neural_model.pt")
        print("  Saved neural_model.pt.")

    print("\n" + "=" * 60)
    print("ALL MODELS SUCCESSFULLY RETRAINED ON TRAIN + VALIDATION DATA!")
    print("=" * 60)

    # 13. Generate Final Submission on File 3 (Test Set)
    print("\n--- Phase 13: Generating Final Submission on Test Set [3] ---")
    sub_output = Path(__file__).parent / "submission_retrained.csv"
    predict_script = Path(__file__).parent / "predict.py"

    cmd = [
        sys.executable,
        str(predict_script),
        "--artifact-dir",
        str(out_dir),
        "--output-csv",
        str(sub_output),
    ]
    print(f"Running inference: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Copy to submissions/ and forecasting_optimized/submission_optimized.csv
    submissions_dir = repo / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sub_output, submissions_dir / "submission_retrained.csv")
    shutil.copy(sub_output, Path(__file__).parent / "submission_optimized.csv")

    df_sub = pd.read_csv(sub_output)
    print("\n" + "=" * 60)
    print("FINAL SUBMISSION FILE CREATED SUCCESSFULLY!")
    print(f"Path: {sub_output}")
    print(f"Total Rows: {len(df_sub):,}")
    print("Class Distribution:")
    print(df_sub["predicted_next_recurring_merchant"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
