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


def assign_stratified_hash_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_column: str = "flow_id",
    stratify_column: str = "source_label",
) -> pd.DataFrame:
    """Assign stable group-wise splits within each label stratum.

    Groups are sorted by a stable hash inside each ``stratify_column`` value and
    then sliced by ratio. This keeps all rows for one group together while
    preserving rare label families across train/val/test when enough groups
    exist.
    """

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train_ratio > 0, val_ratio >= 0, sum < 1")
    if group_column not in frame.columns:
        raise KeyError(f"missing group_column: {group_column}")
    if stratify_column not in frame.columns:
        raise KeyError(f"missing stratify_column: {stratify_column}")

    output = frame.copy()
    group_frame = (
        output[[group_column, stratify_column]]
        .drop_duplicates(subset=[group_column])
        .copy()
    )
    group_frame["_bucket"] = [
        stable_bucket(f"{stratum}:{group}")
        for stratum, group in zip(
            group_frame[stratify_column],
            group_frame[group_column],
            strict=True,
        )
    ]

    group_to_split: dict[str, str] = {}
    for _, stratum_groups in group_frame.groupby(stratify_column, sort=False):
        ordered = stratum_groups.sort_values(["_bucket", group_column], kind="mergesort")
        count = len(ordered)
        if count == 1:
            split_names = ["train"]
        elif count == 2:
            split_names = ["train", "test"]
        else:
            train_count = max(1, int(count * train_ratio))
            val_count = max(1, int(count * val_ratio))
            test_count = count - train_count - val_count
            while test_count < 1 and train_count > 1:
                train_count -= 1
                test_count += 1
            while test_count < 1 and val_count > 1:
                val_count -= 1
                test_count += 1
            train_count += count - (train_count + val_count + test_count)
            split_names = (
                ["train"] * train_count
                + ["val"] * val_count
                + ["test"] * test_count
            )
        for group, split_name in zip(ordered[group_column], split_names, strict=True):
            group_to_split[str(group)] = split_name

    output["split"] = [group_to_split[str(value)] for value in output[group_column]]
    return output


def _flatten_minor_labels(values: pd.Series) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value is None:
            continue
        labels.extend(str(label) for label in value)
    return labels


def processed_stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "flows": 0}
    stats = {
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()),
        "splits": dict(Counter(frame.get("split", []))),
        "views": dict(Counter(frame.get("view", []))),
        "major_labels": dict(Counter(frame.get("major_label", []))),
        "minor_labels": dict(Counter(_flatten_minor_labels(frame["minor_labels"])))
        if "minor_labels" in frame
        else {},
        "source_labels": dict(Counter(frame.get("source_label", []))),
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
