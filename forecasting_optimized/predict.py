from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from blend_models import as_distribution
from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables


def neural_probabilities(
    artifact_dir: Path,
    checkpoint: Path,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    from txembed import (
        PreprocessedTransactions,
        TransactionPreprocessor,
        TransactionSequenceDataset,
        collate_transaction_sequences,
    )

    from neural_model import TemporalTransformerForecastModel

    device = torch.device(device_name)
    preprocessor = TransactionPreprocessor.load(artifact_dir / "preprocessor.json")
    model = TemporalTransformerForecastModel.from_preprocessor(preprocessor).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    rows = PreprocessedTransactions.load_npz(artifact_dir / "test.npz")
    sequences = TransactionSequenceDataset(rows, max_length=max_length)
    loader = DataLoader(
        sequences,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transaction_sequences,
    )
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            output.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    return dict(zip(sequences.client_ids, np.concatenate(output)))


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a submission from the optimized ensemble")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=repository / "data" / "dataset_features" / "test_features.csv",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=repository / "transaction_embedding" / "artifacts" / "uncleaned",
    )
    parser.add_argument(
        "--submission-template",
        type=Path,
        default=repository / "data" / "dataset" / "sample_submission.csv",
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument(
        "--output-csv", type=Path, default=Path(__file__).parent / "submission_optimized.csv"
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neural-cache-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--embedding-pool-cache-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    neural_cache = args.artifact_dir / "neural_test_probabilities.npz"
    embedding_cache = args.artifact_dir / "embedding_pool_test_features.npz"
    if args.neural_cache_only:
        values = neural_probabilities(
            args.embedding_dir,
            args.artifact_dir / "neural_model.pt",
            args.device,
            args.batch_size,
            args.max_length,
        )
        np.savez_compressed(
            neural_cache,
            client_ids=np.asarray(list(values), dtype=str),
            probabilities=np.stack(list(values.values())),
        )
        return
    if args.embedding_pool_cache_only:
        from txembed import TransactionPreprocessor

        from embedding_pool_experiment import aggregate_clients

        preprocessor = TransactionPreprocessor.load(args.embedding_dir / "preprocessor.json")
        clients, features = aggregate_clients(args.embedding_dir, "test", preprocessor)
        np.savez_compressed(embedding_cache, client_ids=clients, features=features)
        return

    # Keep XGBoost imports out of the PyTorch/txembed worker processes. Their
    # native OpenMP runtimes conflict on some macOS installations.
    from ranking_experiment import softmax
    from stacking_experiment import meta_features

    template = pd.read_csv(args.submission_template, dtype={"client_id": str})
    client_ids = template["client_id"]
    frame = pd.read_csv(args.feature_csv)
    wide, pairs = build_feature_tables(frame, client_ids)

    bundle = joblib.load(args.artifact_dir / "model.joblib")
    wide = wide.reindex(columns=bundle["wide_columns"]).fillna(0.0)
    pairs = pairs.reindex(columns=bundle["pair_columns"]).fillna(0.0)
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

    family_bundle = joblib.load(args.artifact_dir / "family_models.joblib")
    family_scores = np.zeros((len(client_ids), len(FAMILIES)), dtype=np.float64)
    for column, family_name in enumerate(FAMILIES):
        family_features = pairs.xs(family_name, level="family").reindex(client_ids)
        family_features = family_features.reindex(columns=family_bundle["columns"])
        family_scores[:, column] = family_bundle["models"][family_name].predict_proba(
            family_features
        )[:, 1]
    family = as_distribution(family_scores)

    if not neural_cache.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--embedding-dir",
                str(args.embedding_dir),
                "--artifact-dir",
                str(args.artifact_dir),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
                "--neural-cache-only",
            ],
            check=True,
        )
    with np.load(neural_cache) as data:
        neural_by_client = dict(zip(data["client_ids"].astype(str), data["probabilities"]))
    neural = np.stack([neural_by_client[client_id] for client_id in client_ids])

    configuration = json.loads((args.artifact_dir / "blend_config.json").read_text())
    weights = configuration["weights"]
    probabilities = (
        weights["catboost"] * catboost
        + weights["pair"] * pair
        + weights["family"] * family
        + weights["neural"] * neural
    )

    extra_sources: dict[str, np.ndarray] = {}
    requested_sources = set(configuration.get("extra_weights", {})) | {
        str(item["source"]) for item in configuration.get("class_adjustments", [])
    }
    if "stacking" in requested_sources:
        stacking = joblib.load(args.artifact_dir / "stacking_model.joblib")
        stacking_pairs = pairs.reindex(columns=stacking["pair_columns"]).fillna(0.0)
        stacking_wide = wide.reindex(columns=stacking["wide_columns"]).fillna(0.0)
        fold_pairs = [
            pair_probabilities(model, stacking_pairs).reindex(client_ids).to_numpy()
            for model in stacking["pair_models"]
        ]
        fold_multiclass = [
            probabilities_in_label_order(model, stacking_wide)
            for model in stacking["catboost_models"]
        ]
        extra_sources["stacking"] = stacking["meta_model"].predict_proba(
            meta_features(np.mean(fold_pairs, axis=0), np.mean(fold_multiclass, axis=0))
        )

    if {"proxy_catboost", "proxy_pair"} & requested_sources:
        proxy = joblib.load(args.artifact_dir / "proxy_pretrained_models.joblib")
        proxy_wide = wide.reindex(columns=proxy["wide_columns"])
        proxy_pairs = pairs.reindex(columns=proxy["pair_columns"])
        extra_sources["proxy_catboost"] = probabilities_in_label_order(
            proxy["catboost_model"], proxy_wide
        )
        extra_sources["proxy_pair"] = as_distribution(
            pair_probabilities(proxy["pair_model"], proxy_pairs).reindex(client_ids).to_numpy()
        )

    if "xgboost_pair" in requested_sources:
        xgboost_bundle = joblib.load(args.artifact_dir / "xgboost_pair_model.joblib")
        xgboost_pairs = pairs.reindex(columns=xgboost_bundle["columns"]).fillna(0.0)
        extra_sources["xgboost_pair"] = as_distribution(
            pair_probabilities(xgboost_bundle["model"], xgboost_pairs)
            .reindex(client_ids)
            .to_numpy()
        )

    if "ranking" in requested_sources:
        ranking = joblib.load(args.artifact_dir / "ranking_model.joblib")
        ranking_pairs = pairs.reindex(columns=ranking["pair_columns"]).fillna(0.0)
        ranking_wide = wide.reindex(columns=ranking["wide_columns"]).fillna(0.0)
        scores = ranking["ranker"].predict(ranking_pairs).reshape(len(client_ids), len(FAMILIES))
        conditional = softmax(scores, ranking["temperature"])
        none_probability = ranking["none_model"].predict_proba(ranking_wide)[:, 1]
        extra_sources["ranking"] = np.column_stack(
            [conditional * (1 - none_probability[:, None]), none_probability]
        )

    if "embedding_pool" in requested_sources:
        if not embedding_cache.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--embedding-dir",
                    str(args.embedding_dir),
                    "--artifact-dir",
                    str(args.artifact_dir),
                    "--embedding-pool-cache-only",
                ],
                check=True,
            )
        with np.load(embedding_cache) as data:
            pooled_by_client = dict(
                zip(data["client_ids"].astype(str), data["features"])
            )
        pooled = np.ascontiguousarray(
            np.stack([pooled_by_client[client_id] for client_id in client_ids]),
            dtype=np.float32,
        )
        embedding_model = joblib.load(args.artifact_dir / "embedding_pool_model.joblib")
        extra_sources["embedding_pool"] = embedding_model["model"].predict_proba(pooled)

    if "catboost_tuned" in requested_sources:
        cat_bundle = joblib.load(args.artifact_dir / "catboost_tuned_bundle.joblib")
        frame["clean_description"] = frame.get("clean_description", pd.Series(dtype=str)).fillna("").astype(str)
        text = frame.groupby("client_id")["clean_description"].apply(lambda texts: " ".join(texts))
        text = text.reindex(client_ids, fill_value="")
        
        test_tfidf = cat_bundle["tfidf"].transform(text)
        test_svd = cat_bundle["svd"].transform(test_tfidf)
        
        tuned_wide = wide.copy()
        for i in range(test_svd.shape[1]):
            tuned_wide[f"tfidf_svd_{i}"] = pd.Series(test_svd[:, i], index=client_ids)
            
        tuned_wide = tuned_wide.reindex(columns=cat_bundle["model"].feature_names_)
        extra_sources["catboost_tuned"] = probabilities_in_label_order(
            cat_bundle["model"], tuned_wide
        )

    if "description_stream" in requested_sources:
        from description_stream_experiment import description_features

        description_bundle = joblib.load(args.artifact_dir / "description_stream_model.joblib")
        description_pairs = pairs.join(description_features(frame, client_ids))
        description_pairs = description_pairs.reindex(columns=description_bundle["columns"]).fillna(0.0)
        extra_sources["description_stream"] = as_distribution(
            pair_probabilities(description_bundle["model"], description_pairs)
            .reindex(client_ids)
            .to_numpy()
        )

    if "micro" in requested_sources:
        micro_bundle = joblib.load(args.artifact_dir / "micro_model.joblib")
        from micro_classifier_experiment import aggregate_text as agg_micro_text
        micro_texts = agg_micro_text(args.feature_csv).reindex(client_ids, fill_value="")
        raw_probs = micro_bundle["model"].predict_proba(micro_texts)
        micro_probs = np.zeros((len(micro_texts), len(LABELS)), dtype=np.float64)
        for i, cls in enumerate(micro_bundle["model"].classes_):
            micro_probs[:, cls] = raw_probs[:, i]
        extra_sources["micro"] = micro_probs

    if "tfidf" in requested_sources:
        tfidf_bundle = joblib.load(args.artifact_dir / "tfidf_model.joblib")
        from tfidf_experiment import aggregate_text as agg_tfidf_text
        tfidf_texts = agg_tfidf_text(args.feature_csv).reindex(client_ids, fill_value="")
        extra_sources["tfidf"] = tfidf_bundle["model"].predict_proba(tfidf_texts)

    for source_name, source_weight in configuration.get("extra_weights", {}).items():
        probabilities = np.exp(
            (1 - source_weight) * np.log(probabilities + 1e-7)
            + source_weight * np.log(extra_sources[source_name] + 1e-7)
        )
        probabilities /= probabilities.sum(axis=1, keepdims=True)

    if "micro_override_weight" in configuration:
        micro_bundle = joblib.load(args.artifact_dir / "micro_model.joblib")
        from micro_classifier_experiment import aggregate_text
        micro_texts = aggregate_text(args.feature_csv).reindex(client_ids, fill_value="")
        raw_probs = micro_bundle["model"].predict_proba(micro_texts)
        micro_probs = np.zeros((len(micro_texts), len(LABELS)), dtype=np.float64)
        for i, cls in enumerate(micro_bundle["model"].classes_):
            micro_probs[:, cls] = raw_probs[:, i]
            
        w = configuration["micro_override_weight"]
        targets = [LABELS.index('music'), LABELS.index('streaming'), LABELS.index('software')]
        for t in targets:
            probabilities[:, t] = np.exp(
                (1 - w) * np.log(probabilities[:, t] + 1e-7) + w * np.log(micro_probs[:, t] + 1e-7)
            )
        probabilities /= probabilities.sum(axis=1, keepdims=True)

    for adjustment in configuration.get("class_adjustments", []):
        column = LABELS.index(str(adjustment["label"]))
        source_weight = float(adjustment["weight"])
        source = extra_sources[str(adjustment["source"])]
        probabilities[:, column] = (
            (1 - source_weight) * probabilities[:, column]
            + source_weight * source[:, column]
        )
        probabilities = np.clip(probabilities, 1e-7, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

    offsets = np.asarray([configuration["class_offsets"][label] for label in LABELS])
    predicted = (
        np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets
    ).argmax(axis=1)
    template["predicted_next_recurring_merchant"] = np.asarray(LABELS)[predicted]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(template):,} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
