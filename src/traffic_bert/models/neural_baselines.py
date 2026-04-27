"""Lightweight neural baselines for byte-token traffic classification."""

from __future__ import annotations

import torch
from torch import nn


def _flatten_windows(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.dim() == 3:
        batch_size, num_windows, seq_len = input_ids.shape
        input_ids = input_ids.reshape(batch_size, num_windows * seq_len)
        attention_mask = attention_mask.reshape(batch_size, num_windows * seq_len)
    return input_ids, attention_mask


class ByteCnnClassifier(nn.Module):
    """1D-CNN baseline over byte token ids."""

    def __init__(self, vocab_size: int, num_labels: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            for kernel_size in (3, 5, 7)
        )
        self.classifier = nn.Linear(hidden_size * len(self.convs), num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = _flatten_windows(input_ids, attention_mask)
        emb = self.embedding(input_ids).transpose(1, 2)
        features = []
        for conv in self.convs:
            out = torch.relu(conv(emb))
            out = out.masked_fill(~attention_mask.bool().unsqueeze(1), torch.finfo(out.dtype).min)
            features.append(out.max(dim=-1).values)
        return self.classifier(torch.cat(features, dim=-1))


class ByteGruClassifier(nn.Module):
    """BiGRU baseline over byte token ids."""

    def __init__(self, vocab_size: int, num_labels: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = _flatten_windows(input_ids, attention_mask)
        emb = self.embedding(input_ids)
        output, _ = self.gru(emb)
        weights = attention_mask.to(dtype=output.dtype).unsqueeze(-1)
        pooled = (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


class ByteTransformerClassifier(nn.Module):
    """Small Transformer encoder baseline without BERT pretraining heads."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = _flatten_windows(input_ids, attention_mask)
        input_ids = input_ids[:, : self.max_length]
        attention_mask = attention_mask[:, : self.max_length]
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        emb = self.embedding(input_ids) + self.position(positions)
        encoded = self.encoder(emb, src_key_padding_mask=~attention_mask.bool())
        weights = attention_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)

