from __future__ import annotations

import math
import torch
from torch import nn
from txembed import TransactionBatch, TransactionEncoder, TransactionPreprocessor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TemporalTransformerForecastModel(nn.Module):
    """Transformer-based order-aware client classifier for the existing transaction embeddings."""

    def __init__(
        self,
        encoder: TransactionEncoder,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        
        # Project raw embedding dimension to hidden_size
        self.input_projection = nn.Linear(TransactionEncoder.OUTPUT_DIM, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    @classmethod
    def from_preprocessor(
        cls, preprocessor: TransactionPreprocessor, **kwargs: object
    ) -> TemporalTransformerForecastModel:
        return cls(TransactionEncoder.from_preprocessor(preprocessor), **kwargs)

    def forward(self, batch: TransactionBatch) -> torch.Tensor:
        embeddings, padding_mask = self.encoder(batch)
        
        # Project and add positional encoding
        x = self.input_projection(embeddings)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Attention over time steps
        attention_logits = self.attention(output).squeeze(-1)
        attention_logits = attention_logits.masked_fill(padding_mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=1)
        attended = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        valid = (~padding_mask).unsqueeze(-1)
        
        # Mean pooling
        mean = (output * valid).sum(dim=1) / lengths.unsqueeze(-1)
        
        # Last element pooling
        last_indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        last = output.gather(1, last_indices).squeeze(1)
        
        return self.classifier(torch.cat([attended, mean, last], dim=-1))
