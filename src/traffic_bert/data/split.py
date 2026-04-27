"""Dataset splitting and processed-data statistics."""

from __future__ import annotations

from collections import Counter
import hashlib

import pandas as pd


def stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def assign_file_time_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_column: str = "source_file",
) -> pd.DataFrame:
    """Assign split by source file/group using a stable hash.

    All rows with the same ``group_column`` value receive the same split, which
    avoids placing nearby flows from the same PCAP into different sets.
    """

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train_ratio > 0, val_ratio >= 0, sum < 1")
    output = frame.copy()
    train_cutoff = int(train_ratio * 10_000)
    val_cutoff = int((train_ratio + val_ratio) * 10_000)

    def choose_split(group: str) -> str:
        bucket = stable_bucket(str(group))
        if bucket < train_cutoff:
            return "train"
        if bucket < val_cutoff:
            return "val"
        return "test"

    output["split"] = [choose_split(value) for value in output[group_column]]
    return output


def processed_stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "flows": 0}
    stats = {
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()),
        "splits": dict(Counter(frame.get("split", []))),
        "views": dict(Counter(frame.get("view", []))),
        "major_labels": dict(Counter(frame.get("major_label", []))),
        "source_datasets": dict(Counter(frame.get("source_dataset", []))),
    }
    for column in ["packet_count", "payload_byte_length", "packet_byte_length"]:
        if column in frame:
            stats[column] = {
                "min": int(frame[column].min()),
                "median": float(frame[column].median()),
                "max": int(frame[column].max()),
            }
    return stats

