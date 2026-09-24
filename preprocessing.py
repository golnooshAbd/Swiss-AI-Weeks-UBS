from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import DescriptionEmbeddingCache
from .schema import FeatureSchema


@dataclass
class PreprocessedTransactions:
    description_embeddings: np.ndarray
    categorical_indices: np.ndarray
    dense_features: np.ndarray
    client_ids: np.ndarray
    timestamps_ns: np.ndarray

    def __len__(self) -> int:
        return len(self.client_ids)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            description_embeddings=self.description_embeddings,
            categorical_indices=self.categorical_indices,
            dense_features=self.dense_features,
            client_ids=self.client_ids.astype(str),
            timestamps_ns=self.timestamps_ns,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "PreprocessedTransactions":
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**{name: arrays[name] for name in cls.__dataclass_fields__})


class TransactionPreprocessor:
    """Fits train-only tabular state and transforms every split identically."""

    def __init__(self, schema: FeatureSchema | None = None) -> None:
        self.schema = schema or FeatureSchema()
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.means is not None and self.stds is not None and bool(self.vocabularies)

    @property
    def categorical_cardinalities(self) -> dict[str, int]:
        self._require_fitted()
        # +1 leaves index zero for missing/unseen values.
        return {name: len(vocabulary) + 1 for name, vocabulary in self.vocabularies.items()}

    @property
    def dense_dimension(self) -> int:
        return len(self.schema.dense_feature_names)

    def fit(self, train: pd.DataFrame) -> "TransactionPreprocessor":
        """Fit only on the training split."""
        self._validate_columns(train)
        if train.empty:
            raise ValueError("Cannot fit preprocessing on an empty training frame")
        self.vocabularies = {}
        for name in self.schema.categorical:
            values = self._categorical_values(train[name])
            unique = sorted(value for value in values.unique().tolist() if value)
            self.vocabularies[name] = {value: index + 1 for index, value in enumerate(unique)}

        continuous, _ = self._continuous_values(train)
        self.means = continuous.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.stds = continuous.std(axis=0, dtype=np.float64).astype(np.float32)
        self.stds[self.stds < 1e-8] = 1.0
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        description_cache: DescriptionEmbeddingCache,
    ) -> PreprocessedTransactions:
        self._require_fitted()
        self._validate_columns(frame)

        if frame[self.schema.client_id].isna().any():
            raise ValueError("client_id contains missing values")

        timestamps = pd.to_datetime(frame[self.schema.timestamp], utc=True, errors="raise")
        if timestamps.isna().any():
            raise ValueError("timestamp contains missing values")

        descriptions = frame[self.schema.description].fillna("").tolist()
        description_embeddings = description_cache.encode(descriptions)

        categorical = np.column_stack(
            [
                self._categorical_values(frame[name])
                .map(self.vocabularies[name])
                .fillna(0)
                .to_numpy(dtype=np.int64)
                for name in self.schema.categorical
            ]
        )

        continuous, missing_flags = self._continuous_values(frame)
        standardized = (continuous - self.means) / self.stds
        boolean = self._boolean_values(frame[self.schema.boolean])[:, None]
        cyclic = self._cyclic_values(frame)
        dense = np.concatenate([boolean, standardized, cyclic, missing_flags], axis=1).astype(np.float32)

        if dense.shape[1] != self.dense_dimension:
            raise RuntimeError(f"Dense feature mismatch: expected {self.dense_dimension}, got {dense.shape[1]}")

        return PreprocessedTransactions(
            description_embeddings=description_embeddings,
            categorical_indices=categorical,
            dense_features=dense,
            client_ids=frame[self.schema.client_id].astype(str).to_numpy(),
            timestamps_ns=timestamps.astype("int64").to_numpy(),
        )

    def fit_transform(
        self,
        train: pd.DataFrame,
        description_cache: DescriptionEmbeddingCache,
    ) -> PreprocessedTransactions:
        return self.fit(train).transform(train, description_cache)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "schema": self.schema.to_dict(),
            "vocabularies": self.vocabularies,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TransactionPreprocessor":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported preprocessing artifact version: {payload.get('version')}")
        instance = cls(FeatureSchema.from_dict(payload["schema"]))
        instance.vocabularies = payload["vocabularies"]
        instance.means = np.asarray(payload["means"], dtype=np.float32)
        instance.stds = np.asarray(payload["stds"], dtype=np.float32)
        return instance

    def _continuous_values(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = {
            name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            for name in self.schema.continuous
        }
        required_numeric = set(self.schema.continuous) - set(self.schema.history)
        missing_required = [name for name in required_numeric if np.isnan(raw[name]).any()]
        if missing_required:
            raise ValueError(
                "Missing or non-numeric values are only supported for history features; "
                f"invalid columns: {sorted(missing_required)}"
            )
        missing_flags = np.column_stack([np.isnan(raw[name]) for name in self.schema.history]).astype(np.float32)

        columns: list[np.ndarray] = []
        for name in self.schema.continuous:
            values = np.nan_to_num(raw[name], nan=0.0)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains an infinite value")
            if name in self.schema.log_continuous:
                if (values < 0).any():
                    raise ValueError(f"{name} must be non-negative before log1p")
                values = np.log1p(values)
            columns.append(values)
        return np.column_stack(columns).astype(np.float32), missing_flags

    def _cyclic_values(self, frame: pd.DataFrame) -> np.ndarray:
        output: list[np.ndarray] = []
        for name, period, offset in zip(
            self.schema.calendar,
            self.schema.calendar_periods,
            self.schema.calendar_offsets,
        ):
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains a missing or infinite value")
            zero_based = values - offset
            if ((zero_based < 0) | (zero_based >= period)).any():
                raise ValueError(
                    f"{name} must be in [{offset}, {offset + period - 1}] for cyclic encoding"
                )
            angle = 2.0 * np.pi * zero_based / period
            output.extend((np.sin(angle), np.cos(angle)))
        return np.column_stack(output).astype(np.float32)

    @staticmethod
    def _categorical_values(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("").str.strip()

    @staticmethod
    def _boolean_values(series: pd.Series) -> np.ndarray:
        mapping = {
            "0": 0.0,
            "0.0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "1": 1.0,
            "1.0": 1.0,
            "true": 1.0,
            "yes": 1.0,
        }
        normalized = series.astype("string").str.strip().str.lower()
        result = normalized.map(mapping)
        if result.isna().any():
            bad = sorted(normalized[result.isna()].dropna().unique().tolist())
            raise ValueError(f"is_recurring_candidate has invalid values: {bad}")
        return result.to_numpy(dtype=np.float32)

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.schema.required_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() on training data or load() an artifact before transform()")
