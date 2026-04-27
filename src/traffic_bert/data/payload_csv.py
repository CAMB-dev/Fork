"""Build processed datasets from Payload-Byte style packet CSV files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_bert.labels import LabelMap


PROCESSED_PAYLOAD_SCHEMA = pa.schema(
    [
        ("flow_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_label", pa.string()),
        ("major_label", pa.string()),
        ("minor_labels", pa.list_(pa.string())),
        ("split", pa.string()),
        ("view", pa.string()),
        ("protocol", pa.string()),
        ("endpoint_a", pa.string()),
        ("endpoint_b", pa.string()),
        ("packet_count", pa.int32()),
        ("payload_byte_length", pa.int32()),
        ("packet_byte_length", pa.int32()),
        ("has_payload", pa.bool_()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("packet_directions", pa.list_(pa.string())),
        ("packet_lengths", pa.list_(pa.int32())),
        ("bytes", pa.binary()),
        ("ttl", pa.int32()),
        ("total_len", pa.int32()),
        ("t_delta", pa.float64()),
    ]
)


@dataclass(frozen=True)
class PayloadCsvBuildConfig:
    input_path: Path
    output_path: Path
    label_map_path: Path
    source_dataset: str
    split: str = "train"
    label_column: str = "label"
    byte_prefix: str = "payload_byte_"
    chunksize: int = 20_000
    keep_empty_payload: bool = False
    trim_trailing_zeros: bool = True
    compression: str = "zstd"


def _byte_columns(columns: list[str], prefix: str) -> list[str]:
    selected = [column for column in columns if column.startswith(prefix)]
    if not selected:
        raise ValueError(f"no byte columns found with prefix {prefix!r}")
    return sorted(selected, key=lambda value: int(value.removeprefix(prefix)))


def _trimmed_lengths(values: np.ndarray) -> np.ndarray:
    nonzero = values != 0
    has_payload = nonzero.any(axis=1)
    reversed_last = np.argmax(nonzero[:, ::-1], axis=1)
    lengths = values.shape[1] - reversed_last
    return np.where(has_payload, lengths, 0).astype(np.int32)


def _full_lengths(values: np.ndarray) -> np.ndarray:
    return np.full(values.shape[0], values.shape[1], dtype=np.int32)


def _protocol_values(chunk: pd.DataFrame) -> list[str]:
    if "protocol" not in chunk:
        return ["unknown"] * len(chunk)
    return [str(value).lower() for value in chunk["protocol"]]


def _optional_numeric(chunk: pd.DataFrame, column: str, default: int | float) -> list[Any]:
    if column not in chunk:
        return [default] * len(chunk)
    return chunk[column].fillna(default).tolist()


def _write_chunk(
    writer: pq.ParquetWriter | None,
    frame: pd.DataFrame,
    output_path: Path,
    compression: str,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(
        frame,
        schema=PROCESSED_PAYLOAD_SCHEMA,
        preserve_index=False,
    )
    if writer is None:
        writer = pq.ParquetWriter(output_path, PROCESSED_PAYLOAD_SCHEMA, compression=compression)
    writer.write_table(table)
    return writer


def build_payload_csv_dataset(config: PayloadCsvBuildConfig) -> dict[str, Any]:
    """Convert a Payload-Byte CSV into the common processed Parquet schema."""

    label_map = LabelMap.from_yaml(config.label_map_path)
    header = pd.read_csv(config.input_path, nrows=0)
    byte_columns = _byte_columns(list(header.columns), config.byte_prefix)
    required_columns = set(byte_columns) | {config.label_column}
    missing = sorted(required_columns - set(header.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.output_path.exists():
        config.output_path.unlink()

    usecols = byte_columns + [
        column
        for column in ["ttl", "total_len", "protocol", "t_delta", config.label_column]
        if column in header.columns
    ]
    dtype = {column: "uint8" for column in byte_columns}
    if "ttl" in usecols:
        dtype["ttl"] = "uint16"
    if "total_len" in usecols:
        dtype["total_len"] = "uint16"

    writer: pq.ParquetWriter | None = None
    stats: dict[str, Any] = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "source_dataset": config.source_dataset,
        "byte_columns": len(byte_columns),
        "rows_read": 0,
        "rows_written": 0,
        "dropped_empty_payload": 0,
    }
    source_labels: Counter[str] = Counter()
    major_labels: Counter[str] = Counter()
    minor_labels: Counter[str] = Counter()
    protocols: Counter[str] = Counter()
    length_hist = np.zeros(len(byte_columns) + 1, dtype=np.int64)

    try:
        reader = pd.read_csv(
            config.input_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=config.chunksize,
        )
        row_offset = 0
        for chunk in reader:
            values = chunk[byte_columns].to_numpy(dtype=np.uint8, copy=False)
            lengths = (
                _trimmed_lengths(values)
                if config.trim_trailing_zeros
                else _full_lengths(values)
            )
            keep_mask = np.ones(len(chunk), dtype=bool)
            if not config.keep_empty_payload:
                keep_mask = lengths > 0

            kept_indices = np.flatnonzero(keep_mask)
            dropped = len(chunk) - len(kept_indices)
            labels = [str(value) for value in chunk[config.label_column]]
            targets = [label_map.map_source_label(label) for label in labels]
            protocol_values = _protocol_values(chunk)
            ttl_values = _optional_numeric(chunk, "ttl", 0)
            total_len_values = _optional_numeric(chunk, "total_len", 0)
            t_delta_values = _optional_numeric(chunk, "t_delta", 0.0)

            rows: list[dict[str, Any]] = []
            for idx in kept_indices:
                source_label = labels[idx]
                target = targets[idx]
                payload_length = int(lengths[idx])
                payload = values[idx, :payload_length].tobytes()
                global_index = row_offset + int(idx)
                protocols[protocol_values[idx]] += 1
                source_labels[source_label] += 1
                major_labels[target.major_label] += 1
                minor_labels.update(target.minor_labels)
                length_hist[payload_length] += 1
                rows.append(
                    {
                        "flow_id": f"{config.source_dataset}:packet:{global_index}",
                        "source_dataset": config.source_dataset,
                        "source_file": str(config.input_path),
                        "source_label": source_label,
                        "major_label": target.major_label,
                        "minor_labels": list(target.minor_labels),
                        "split": config.split,
                        "view": "payload_only",
                        "protocol": protocol_values[idx],
                        "endpoint_a": "unknown:0",
                        "endpoint_b": "unknown:0",
                        "packet_count": 1,
                        "payload_byte_length": payload_length,
                        "packet_byte_length": payload_length,
                        "has_payload": payload_length > 0,
                        "start_time": float(global_index),
                        "end_time": float(global_index),
                        "packet_directions": ["fwd"],
                        "packet_lengths": [payload_length],
                        "bytes": payload,
                        "ttl": int(ttl_values[idx]),
                        "total_len": int(total_len_values[idx]),
                        "t_delta": float(t_delta_values[idx]),
                    }
                )

            if rows:
                writer = _write_chunk(
                    writer=writer,
                    frame=pd.DataFrame(rows),
                    output_path=config.output_path,
                    compression=config.compression,
                )

            stats["rows_read"] += int(len(chunk))
            stats["rows_written"] += int(len(rows))
            stats["dropped_empty_payload"] += int(dropped)
            row_offset += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame().to_parquet(config.output_path, index=False)

    nonzero_lengths = np.flatnonzero(length_hist)
    stats.update(
        {
            "source_labels": dict(source_labels),
            "major_labels": dict(major_labels),
            "minor_labels": dict(minor_labels),
            "protocols": dict(protocols),
            "payload_byte_length": _length_stats(length_hist, nonzero_lengths),
        }
    )

    stats_path = config.output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def _length_stats(length_hist: np.ndarray, nonzero_lengths: np.ndarray) -> dict[str, int | float]:
    total = int(length_hist.sum())
    if total == 0:
        return {"min": 0, "median": 0.0, "max": 0}

    cumulative = np.cumsum(length_hist)
    midpoint = (total + 1) // 2
    median = int(np.searchsorted(cumulative, midpoint, side="left"))
    return {
        "min": int(nonzero_lengths[0]) if len(nonzero_lengths) else 0,
        "median": float(median),
        "max": int(nonzero_lengths[-1]) if len(nonzero_lengths) else 0,
    }
