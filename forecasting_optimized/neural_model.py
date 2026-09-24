from __future__ import annotations

import torch
from torch import nn
from txembed import TransactionBatch, TransactionEncoder, TransactionPreprocessor


class TemporalGRUForecastModel(nn.Module):
    """Order-aware client classifier for the existing transaction embeddings."""

    def __init__(
        self,
        encoder: TransactionEncoder,
        hidden_size: int = 112,
        dropout: float = 0.2,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(
            input_size=TransactionEncoder.OUTPUT_DIM,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        output_size = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_size * 3),
            nn.Dropout(dropout),
            nn.Linear(output_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    @classmethod
    def from_preprocessor(
        cls, preprocessor: TransactionPreprocessor, **kwargs: object
    ) -> TemporalGRUForecastModel:
        return cls(TransactionEncoder.from_preprocessor(preprocessor), **kwargs)

    def forward(self, batch: TransactionBatch) -> torch.Tensor:
        embeddings, padding_mask = self.encoder(batch)
        lengths = (~padding_mask).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=embeddings.shape[1],
        )

        attention_logits = self.attention(output).squeeze(-1)
        attention_logits = attention_logits.masked_fill(padding_mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        attended = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        valid = (~padding_mask).unsqueeze(-1)
        mean = (output * valid).sum(dim=1) / lengths.unsqueeze(-1)
        last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        last = output.gather(1, last_indices).squeeze(1)
        return self.classifier(torch.cat([attended, mean, last], dim=-1))
