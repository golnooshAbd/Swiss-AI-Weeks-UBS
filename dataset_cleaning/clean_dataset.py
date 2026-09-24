from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")


def clean_text(value: object) -> str:
    """Remove volatile identifiers while retaining the description's semantics."""
    text = str(value).lower()
    text = re.sub(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", " ", text)
    text = re.sub(r"\b\w*\d+\w*\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def classify_transaction(mcc: object, description: object, amount: object) -> str:
    mcc = str(mcc)
    description = str(description).lower()
    amount = float(amount)
    noise = (
        "salary",
        "atm withdrawal",
        "fresh foods",
        "pharmacy",
        "hotel booking",
        "electronics shop",
        "coffee shop",
        "neighborhood market",
        "grocery store",
        "online marketplace",
        "ride share",
        "casual dining",
        "p2p send",
        "p2p receive",
        "service fee",
    )
    if any(term in description for term in noise):
        return "none"
    if mcc == "7997" or any(term in description for term in ("gym", "fitness", "fit club")):
        return "gym"
    if mcc == "6300" or any(
        term in description for term in ("insurance", "safe cover", "policy", "cover plan")
    ):
        return "insurance"
    if mcc == "4814" or any(
        term in description for term in ("phone", "contract", "telecom", "carrier")
    ):
        return "mobile"
    if mcc == "5734":
        return "cloud" if any(term in description for term in ("cloud", "storage", "backup")) else "software"
    if mcc == "5732" and any(
        term in description for term in ("cloud", "storage", "backup", "service plan")
    ):
        return "cloud"
    if mcc == "5812":
        if any(term in description for term in ("audio", "member pass")):
            return "music"
        if any(term in description for term in ("video", "media stream", "streaming", "stream")):
            return "streaming"
    if "digital plus" in description or "premium plan" in description:
        if mcc == "4814":
            return "mobile"
        if mcc == "5734":
            return "software"
        if mcc == "5812":
            return "music" if amount < 16 else "streaming"
    if "monthly plan" in description:
        return {
            "4814": "mobile",
            "6300": "insurance",
            "7997": "gym",
            "5734": "software",
            "5732": "cloud",
            "5812": "streaming",
        }.get(mcc, "none")
    return "none"


def load_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for source_order, line in enumerate(handle):
            record = json.loads(line)
            record["_source_order"] = source_order
            records.append(record)
    return pd.DataFrame.from_records(records)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain every transaction and add leakage-safe temporal/history features."""
    required = {
        "amount",
        "client_id",
        "currency",
        "description",
        "direction",
        "mcc",
        "timestamp",
        "type",
        "_source_order",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    result = frame.copy()
    original_columns = [column for column in result.columns if column != "_source_order"]
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result = result.sort_values(
        ["client_id", "timestamp", "_source_order"], kind="stable"
    ).reset_index(drop=True)

    result["clean_description"] = result["description"].map(clean_text)
    result["candidate_family"] = [
        classify_transaction(mcc, description, amount)
        for mcc, description, amount in zip(
            result["mcc"], result["description"], result["amount"]
        )
    ]
    result["day_of_week"] = result["timestamp"].dt.dayofweek
    result["day_of_month"] = result["timestamp"].dt.day
    result["month"] = result["timestamp"].dt.month
    result["days_to_cutoff"] = (CUTOFF - result["timestamp"]).dt.total_seconds() / 86_400
    result["days_since_prev_transaction"] = (
        result.groupby("client_id", sort=False)["timestamp"].diff().dt.total_seconds() / 86_400
    )

    history_columns = (
        "days_since_prev_same_family",
        "count_same_family_before",
        "median_interval_same_family",
        "std_interval_same_family",
        "median_amount_same_family",
        "amount_vs_family_median",
    )
    for column in history_columns:
        result[column] = np.nan

    candidate = result["candidate_family"].ne("none")
    recurring = result.loc[candidate, ["client_id", "candidate_family", "timestamp", "amount"]].copy()
    if not recurring.empty:
        grouped = recurring.groupby(["client_id", "candidate_family"], sort=False)
        recurring["gap"] = grouped["timestamp"].diff().dt.total_seconds() / 86_400
        recurring["days_since_prev_same_family"] = recurring["gap"]
        recurring["count_same_family_before"] = grouped.cumcount()
        recurring["median_interval_same_family"] = grouped["gap"].transform(
            lambda values: values.expanding().median()
        )
        recurring["std_interval_same_family"] = grouped["gap"].transform(
            lambda values: values.expanding().std()
        )
        recurring["median_amount_same_family"] = grouped["amount"].transform(
            lambda values: values.expanding().median()
        )
        recurring["amount_vs_family_median"] = (
            recurring["amount"] - recurring["median_amount_same_family"]
        )
        result.loc[candidate, list(history_columns)] = recurring[list(history_columns)]

    added_columns = [
        "clean_description",
        "candidate_family",
        "day_of_week",
        "day_of_month",
        "month",
        "days_to_cutoff",
        "days_since_prev_transaction",
        *history_columns,
    ]
    return result[[*original_columns, *added_columns]]


def build_archive(output_dir: Path, input_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for split in SPLITS:
            path = output_dir / f"{split}_features.csv"
            archive.write(path, path.name)
        for name in ("train_labels.csv", "valid_labels.csv", "sample_submission.csv"):
            path = input_dir / name
            if path.exists():
                archive.write(path, name)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Clean transaction splits and package the result")
    parser.add_argument("--input-dir", type=Path, default=repository / "data" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=repository / "data" / "dataset_features")
    parser.add_argument("--archive", type=Path, default=repository / "data" / "dataset_features.zip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        input_path = args.input_dir / f"{split}_transactions.jsonl"
        output_path = args.output_dir / f"{split}_features.csv"
        cleaned = clean_transactions(load_jsonl(input_path))
        cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"{split}: {len(cleaned):,} transactions -> {output_path}")

    build_archive(args.output_dir, args.input_dir, args.archive)
    print(f"archive: {args.archive}")


if __name__ == "__main__":
    main()
