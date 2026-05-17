"""Torch datasets backed by processed Parquet flow rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from traffic_bert.labels import LabelMap
from traffic_bert.sce import SemanticCodebook
from traffic_bert.tokenizer import ByteTokenizer, PacketChunk


CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "context_host_prev_60s_count",
    "context_host_prev_300s_count",
    "context_host_prev_60s_reset_count",
    "context_host_prev_60s_control_count",
    "context_host_prev_60s_payloadless_count",
    "context_host_prev_60s_unique_dst_ports",
    "context_host_prev_60s_unique_dst_hosts",
)


@dataclass(frozen=True)
class FlowExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    window_mask: torch.Tensor
    major_label: torch.Tensor
    minor_labels: torch.Tensor
    flow_id: str
    context_features: torch.Tensor | None = None


def _chunks_from_row(row: pd.Series) -> list[PacketChunk]:
    payload = row["bytes"]
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif not isinstance(payload, bytes):
        payload = bytes(payload)

    lengths = list(row["packet_lengths"])
    directions = list(row["packet_directions"])
    chunks: list[PacketChunk] = []
    offset = 0
    for direction, length in zip(directions, lengths, strict=True):
        next_offset = offset + int(length)
        chunks.append(PacketChunk(direction=direction, data=payload[offset:next_offset]))
        offset = next_offset
    return chunks


class FlowWindowDataset(Dataset):
    """Read processed flow rows and produce padded window tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        label_map: LabelMap,
        tokenizer: ByteTokenizer | None = None,
        max_length: int = 512,
        stride: int = 384,
        max_windows: int | None = None,
        use_connection_tokens: bool = False,
        use_context_tokens: bool = False,
        use_context_features: bool = False,
        semantic_codebook: SemanticCodebook | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_map = label_map
        self.semantic_codebook = semantic_codebook
        self.tokenizer = tokenizer or ByteTokenizer(
            extra_tokens=semantic_codebook.tokens if semantic_codebook is not None else None
        )
        if semantic_codebook is not None:
            missing = [token for token in semantic_codebook.tokens if token not in self.tokenizer.token_to_id]
            if missing:
                raise ValueError(f"tokenizer is missing SCE codebook tokens: {missing[:5]}")
        self.max_length = max_length
        self.stride = stride
        self.max_windows = max_windows
        self.use_connection_tokens = use_connection_tokens
        self.use_context_tokens = use_context_tokens
        self.use_context_features = use_context_features

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        label_map: LabelMap,
        view: str | None = None,
        split: str | None = None,
        **kwargs: Any,
    ) -> "FlowWindowDataset":
        frame = pd.read_parquet(path)
        if view is not None:
            frame = frame[frame["view"] == view]
        if split is not None:
            frame = frame[frame["split"] == split]
        return cls(frame=frame, label_map=label_map, **kwargs)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> FlowExample:
        row = self.frame.iloc[index]
        prefix_tokens = None
        if self.use_connection_tokens and "connection_type" in row:
            prefix_tokens = [self.tokenizer.connection_type_token(row["connection_type"])]
        if self.use_context_tokens:
            prefix_tokens = list(prefix_tokens or [])
            context_columns = [
                ("H60", "context_host_prev_60s_count"),
                ("H300", "context_host_prev_300s_count"),
                ("R60", "context_host_prev_60s_reset_count"),
                ("C60", "context_host_prev_60s_control_count"),
                ("P0_60", "context_host_prev_60s_payloadless_count"),
                ("DPORT60", "context_host_prev_60s_unique_dst_ports"),
                ("DHOST60", "context_host_prev_60s_unique_dst_hosts"),
            ]
            for feature, column in context_columns:
                if column in row:
                    prefix_tokens.append(self.tokenizer.context_token(feature, row[column]))
        byte_token_encoder = None
        if self.semantic_codebook is not None:
            def encode_with_codebook(data: bytes) -> list[str]:
                return self.semantic_codebook.encode_bytes(data, tokenizer=self.tokenizer)

            byte_token_encoder = encode_with_codebook
        encodings = self.tokenizer.encode_flow(
            _chunks_from_row(row),
            max_length=self.max_length,
            stride=self.stride,
            padding=True,
            prefix_tokens=prefix_tokens,
            byte_token_encoder=byte_token_encoder,
        )
        if self.max_windows is not None:
            encodings = encodings[: self.max_windows]

        input_ids = torch.tensor([item.input_ids for item in encodings], dtype=torch.long)
        attention_mask = torch.tensor(
            [item.attention_mask for item in encodings], dtype=torch.long
        )
        window_mask = torch.ones(input_ids.size(0), dtype=torch.bool)
        major_label = torch.tensor(
            self.label_map.major_id(str(row["major_label"])),
            dtype=torch.long,
        )
        minor_raw = row["minor_labels"]
        minor_labels = torch.tensor(
            self.label_map.minor_multi_hot(list(minor_raw)),
            dtype=torch.float32,
        )
        context_features = None
        if self.use_context_features:
            values = []
            for column in CONTEXT_FEATURE_COLUMNS:
                value = row[column] if column in row and pd.notna(row[column]) else 0.0
                values.append(float(value))
            context_features = torch.log1p(torch.tensor(values, dtype=torch.float32))

        return FlowExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            window_mask=window_mask,
            major_label=major_label,
            minor_labels=minor_labels,
            flow_id=str(row["flow_id"]),
            context_features=context_features,
        )


def flow_collate(batch: list[FlowExample]) -> dict[str, Any]:
    max_windows = max(item.input_ids.size(0) for item in batch)
    seq_len = batch[0].input_ids.size(1)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, max_windows, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_windows, seq_len, dtype=torch.long)
    window_mask = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    major_labels = torch.stack([item.major_label for item in batch])
    minor_labels = torch.stack([item.minor_labels for item in batch])
    context_items = [item.context_features for item in batch]

    for idx, item in enumerate(batch):
        n_windows = item.input_ids.size(0)
        input_ids[idx, :n_windows] = item.input_ids
        attention_mask[idx, :n_windows] = item.attention_mask
        window_mask[idx, :n_windows] = item.window_mask

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "window_mask": window_mask,
        "major_labels": major_labels,
        "minor_labels": minor_labels,
        "flow_ids": [item.flow_id for item in batch],
    }
    if all(item is not None for item in context_items):
        result["context_features"] = torch.stack([item for item in context_items if item is not None])
    return result
