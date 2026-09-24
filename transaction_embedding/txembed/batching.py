from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .preprocessing import PreprocessedTransactions


class TransactionSequence(TypedDict):
    description_embeddings: torch.Tensor
    categorical_indices: torch.Tensor
    dense_features: torch.Tensor


@dataclass
class TransactionBatch:
    description_embeddings: torch.Tensor  # [B, T, 384]
    categorical_indices: torch.Tensor  # [B, T, 5]
    dense_features: torch.Tensor  # [B, T, dense_dim]
    padding_mask: torch.Tensor  # [B, T], True where padded

    def to(self, device: torch.device | str) -> "TransactionBatch":
        return TransactionBatch(**{name: value.to(device) for name, value in vars(self).items()})


class TransactionSequenceDataset(Dataset[TransactionSequence]):
    """Groups rows by client and sorts each client's history oldest-first."""

    def __init__(self, data: PreprocessedTransactions, max_length: int | None = None) -> None:
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        self.data = data
        self.max_length = max_length
        groups: dict[str, list[int]] = {}
        for index, client_id in enumerate(data.client_ids):
            groups.setdefault(str(client_id), []).append(index)
        self.client_ids = list(groups)
        self.indices = [
            np.asarray(sorted(indices, key=lambda i: int(data.timestamps_ns[i])), dtype=np.int64)
            for indices in groups.values()
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> TransactionSequence:
        indices = self.indices[index]
        if self.max_length is not None:
            # Retain the most recent context while preserving chronological order.
            indices = indices[-self.max_length :]
        return {
            "description_embeddings": torch.from_numpy(self.data.description_embeddings[indices]),
            "categorical_indices": torch.from_numpy(self.data.categorical_indices[indices]),
            "dense_features": torch.from_numpy(self.data.dense_features[indices]),
        }


def collate_transaction_sequences(sequences: list[TransactionSequence]) -> TransactionBatch:
    if not sequences:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(item["dense_features"]) for item in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    positions = torch.arange(max_length).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    return TransactionBatch(
        description_embeddings=pad_sequence(
            [item["description_embeddings"] for item in sequences], batch_first=True
        ),
        categorical_indices=pad_sequence(
            [item["categorical_indices"] for item in sequences], batch_first=True
        ),
        dense_features=pad_sequence(
            [item["dense_features"] for item in sequences], batch_first=True
        ),
        padding_mask=padding_mask,
    )

