from __future__ import annotations

import torch
from torch import nn

from .batching import TransactionBatch
from .preprocessing import TransactionPreprocessor


class TransactionEncoder(nn.Module):
    """Maps preprocessed transaction sequences to [B, T, 128]."""

    DESCRIPTION_INPUT_DIM = 384
    DESCRIPTION_OUTPUT_DIM = 64
    OUTPUT_DIM = 128

    def __init__(
        self,
        categorical_cardinalities: dict[str, int],
        dense_dimension: int,
        categorical_names: tuple[str, ...] = (
            "candidate_family",
            "mcc",
            "type",
            "currency",
            "direction",
        ),
        categorical_dims: tuple[int, ...] = (8, 8, 6, 4, 2),
    ) -> None:
        super().__init__()
        if set(categorical_cardinalities) != set(categorical_names):
            raise ValueError("Categorical cardinalities do not match the expected feature names")
        self.categorical_names = categorical_names
        self.description_projection = nn.Linear(self.DESCRIPTION_INPUT_DIM, self.DESCRIPTION_OUTPUT_DIM)
        # ModuleList avoids collisions between feature names such as "type" and
        # existing nn.Module methods. Its order is the schema's categorical order.
        self.categorical_embeddings = nn.ModuleList(
            [
                # Index 0 is a learned unknown-category representation. Padded
                # positions also contain zeros, but are masked out after encoding.
                nn.Embedding(categorical_cardinalities[name], dim)
                for name, dim in zip(categorical_names, categorical_dims)
            ]
        )
        input_dimension = self.DESCRIPTION_OUTPUT_DIM + sum(categorical_dims) + dense_dimension
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    @classmethod
    def from_preprocessor(cls, preprocessor: TransactionPreprocessor) -> "TransactionEncoder":
        return cls(
            categorical_cardinalities=preprocessor.categorical_cardinalities,
            dense_dimension=preprocessor.dense_dimension,
            categorical_names=preprocessor.schema.categorical,
            categorical_dims=preprocessor.schema.categorical_dims,
        )

    def forward(self, batch: TransactionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        description = self.description_projection(batch.description_embeddings.float())
        categorical = [
            layer(batch.categorical_indices[..., column].long())
            for column, layer in enumerate(self.categorical_embeddings)
        ]
        combined = torch.cat([description, *categorical, batch.dense_features.float()], dim=-1)
        output = self.network(combined)
        # Make padded output deterministic and harmless to downstream pooling.
        output = output.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        return output, batch.padding_mask
