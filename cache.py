from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class DescriptionEmbeddingCache:
    """Persistent, content-addressed cache for frozen MiniLM embeddings."""

    def __init__(
        self,
        path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: str | None = None,
        encoder_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self._encoder_factory = encoder_factory
        self._encoder: Any | None = None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )"""
        )
        return connection

    def _key(self, text: str) -> str:
        value = f"{self.model_name}\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            if self._encoder_factory is None:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._encoder = self._encoder_factory(self.model_name)
            self._encoder.eval()
            # The encoder is preprocessing-only and never belongs to the trainable model.
            for parameter in self._encoder.parameters():
                parameter.requires_grad_(False)
        return self._encoder

    def encode(self, descriptions: Sequence[object], batch_size: int = 256) -> np.ndarray:
        texts = ["" if value is None else str(value).strip() for value in descriptions]
        unique_texts = list(dict.fromkeys(texts))
        found: dict[str, np.ndarray] = {}

        with self._connect() as connection:
            for text in unique_texts:
                row = connection.execute(
                    "SELECT vector FROM embeddings WHERE cache_key = ?", (self._key(text),)
                ).fetchone()
                if row is not None:
                    vector = np.frombuffer(row[0], dtype=np.float32).copy()
                    if vector.shape != (self.dimension,):
                        raise ValueError(
                            f"Corrupt cached embedding for key {self._key(text)}: {vector.shape}"
                        )
                    found[text] = vector

            missing = [text for text in unique_texts if text not in found]
            if missing:
                encoder = self._load_encoder()
                vectors = np.asarray(
                    encoder.encode(
                        missing,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=len(missing) >= batch_size,
                    ),
                    dtype=np.float32,
                )
                if vectors.shape != (len(missing), self.dimension):
                    raise ValueError(
                        f"Expected description embeddings {(len(missing), self.dimension)}, "
                        f"got {vectors.shape}"
                    )
                connection.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                    [
                        (self._key(text), self.model_name, self.dimension, vector.tobytes())
                        for text, vector in zip(missing, vectors)
                    ],
                )
                found.update(zip(missing, vectors))

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([found[text] for text in texts]).astype(np.float32, copy=False)
