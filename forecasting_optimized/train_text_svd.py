"""
Train an unsupervised TF-IDF + SVD pipeline on family-level transaction descriptions
to extract text features for the Pair Model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REPO = Path(__file__).resolve().parents[1]
FEATURE_DIR = REPO / "data" / "dataset_features"

def main() -> None:
    print("Loading text data for SVD...", flush=True)
    frame = pd.read_csv(FEATURE_DIR / "train_features.csv", usecols=["client_id", "candidate_family", "clean_description"])
    frame["clean_description"] = frame["clean_description"].fillna("").astype(str)
    
    # Aggregate text at the FAMILY level, because the Pair Model extracts features per family
    print("Aggregating text by (client_id, candidate_family)...", flush=True)
    family_texts = frame.groupby(["client_id", "candidate_family"])["clean_description"].apply(lambda texts: " ".join(texts))
    
    print("Training TF-IDF + SVD (10 components)...", flush=True)
    pipeline = make_pipeline(
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        ),
        TruncatedSVD(n_components=10, random_state=2026)
    )
    
    pipeline.fit(family_texts.values)
    
    print(f"Explained variance ratio sum: {pipeline.named_steps['truncatedsvd'].explained_variance_ratio_.sum():.4f}")
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_DIR / "text_svd_pipeline.joblib")
    print("Saved text_svd_pipeline.joblib", flush=True)

if __name__ == "__main__":
    main()
