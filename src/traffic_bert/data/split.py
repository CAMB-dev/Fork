"""Dataset splitting and processed-data statistics."""

from __future__ import annotations

from collections import Counter
import hashlib

import pandas as pd

from traffic_bert.data.split_audit import minor_support_by_split, support_by_split


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


def assign_time_ordered_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_column: str = "flow_id",
    stratify_column: str = "source_label",
    time_column: str = "start_time",
) -> pd.DataFrame:
    """Assign train/val/test by time order within each label stratum.

    Groups are ordered by ``time_column`` within each ``stratify_column`` value,
    then sliced by ratio. This gives a stricter split than random flow hashing
    while still keeping rare labels represented when enough groups exist.
    """

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train_ratio > 0, val_ratio >= 0, sum < 1")
    for column in [group_column, stratify_column, time_column]:
        if column not in frame.columns:
            raise KeyError(f"missing column: {column}")

    output = frame.copy()
    group_frame = (
        output[[group_column, stratify_column, time_column]]
        .sort_values([time_column, group_column], kind="mergesort")
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
        ordered = stratum_groups.sort_values(
            [time_column, "_bucket", group_column],
            kind="mergesort",
        )
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


def assign_time_block_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_column: str = "flow_id",
    stratify_column: str = "source_label",
    time_column: str = "start_time",
    block_size: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign splits by shuffled contiguous time blocks within each label.

    This is a compromise between random flow splitting and strict time-ordered
    splitting: nearby flows stay in the same block, but blocks are distributed
    across train/val/test to cover different time ranges in each split.
    """

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train_ratio > 0, val_ratio >= 0, sum < 1")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for column in [group_column, stratify_column, time_column]:
        if column not in frame.columns:
            raise KeyError(f"missing column: {column}")

    output = frame.copy()
    group_frame = (
        output[[group_column, stratify_column, time_column]]
        .sort_values([stratify_column, time_column, group_column], kind="mergesort")
        .drop_duplicates(subset=[group_column])
        .copy()
    )

    group_to_split: dict[str, str] = {}
    for stratum, stratum_groups in group_frame.groupby(stratify_column, sort=False):
        ordered = stratum_groups.sort_values([time_column, group_column], kind="mergesort")
        count = len(ordered)
        if count == 1:
            group_to_split[str(ordered.iloc[0][group_column])] = "train"
            continue
        if count == 2:
            group_to_split[str(ordered.iloc[0][group_column])] = "train"
            group_to_split[str(ordered.iloc[1][group_column])] = "test"
            continue

        effective_block_size = min(block_size, max(1, count // 3))
        block_ids = [idx // effective_block_size for idx in range(count)]
        tail_size = count % effective_block_size
        if tail_size and count > effective_block_size:
            min_tail_size = max(2, effective_block_size // 2)
            if tail_size < min_tail_size:
                tail_block = block_ids[-1]
                block_ids = [
                    tail_block - 1 if block_id == tail_block else block_id
                    for block_id in block_ids
                ]

        work = ordered.copy()
        work["_block"] = block_ids
        block_frame = (
            work.groupby("_block", sort=True)
            .agg(
                size=(group_column, "size"),
                first_time=(time_column, "min"),
            )
            .reset_index()
        )
        block_frame["_shuffle"] = [
            stable_bucket(f"{seed}:{stratum}:{block_id}")
            for block_id in block_frame["_block"]
        ]
        block_frame = block_frame.sort_values(
            ["_shuffle", "first_time", "_block"],
            kind="mergesort",
        )

        train_target = max(1, int(count * train_ratio))
        val_target = max(1, int(count * val_ratio))
        test_target = count - train_target - val_target
        while test_target < 1 and train_target > 1:
            train_target -= 1
            test_target += 1
        while test_target < 1 and val_target > 1:
            val_target -= 1
            test_target += 1

        block_to_split: dict[int, str] = {}
        assigned = {"train": 0, "val": 0, "test": 0}
        targets = {
            "train": train_target,
            "val": val_target,
            "test": test_target,
        }
        block_rows = list(
            block_frame.itertuples(
                index=False,
                name=None,
            )
        )
        required_splits = ["train", "val", "test"] if len(block_rows) >= 3 else []
        for row_index, (block_id, size, _first_time, _shuffle) in enumerate(block_rows):
            size = int(size)
            if row_index < len(required_splits):
                split_name = required_splits[row_index]
            else:
                deficits = {
                    split: targets[split] - assigned[split]
                    for split in ["train", "val", "test"]
                }
                split_name = max(
                    deficits,
                    key=lambda split: (deficits[split], -assigned[split]),
                )
            block_to_split[int(block_id)] = split_name
            assigned[split_name] += size

        for group, block_id in work[[group_column, "_block"]].itertuples(
            index=False,
            name=None,
        ):
            group_to_split[str(group)] = block_to_split[int(block_id)]

    output["split"] = [group_to_split[str(value)] for value in output[group_column]]
    return output


def stratified_sample(
    frame: pd.DataFrame,
    stratify_column: str = "major_label",
    max_per_class: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic per-class capped sample.

    A non-positive cap means "no cap"; formal full-dataset splits use that
    mode so preprocessing does not silently downsample otherwise complete data.
    """

    if stratify_column not in frame.columns:
        raise KeyError(f"missing stratify_column: {stratify_column}")
    if max_per_class <= 0:
        output = frame.copy()
        if {"start_time", "flow_id"} <= set(frame.columns):
            output = output.sort_values(["start_time", "flow_id"], kind="mergesort")
        return output

    parts = []
    for _, group in frame.groupby(stratify_column, sort=True):
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=seed)
        parts.append(group)
    if not parts:
        return frame.copy()
    output = pd.concat(parts, ignore_index=True)
    if {"start_time", "flow_id"} <= set(frame.columns):
        output = output.sort_values(["start_time", "flow_id"], kind="mergesort")
    return output


def _flatten_minor_labels(values: pd.Series) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value is None:
            continue
        labels.extend(str(label) for label in value)
    return labels


def processed_stats(frame: pd.DataFrame, metadata: dict | None = None) -> dict:
    if frame.empty:
        stats = {"rows": 0, "flows": 0}
        if metadata:
            stats.update(metadata)
        return stats
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
        "connection_types": dict(Counter(frame["connection_type"]))
        if "connection_type" in frame
        else {},
    }
    if "was_packet_truncated" in frame:
        stats["packet_truncated_rows"] = int(frame["was_packet_truncated"].sum())
        stats["packet_truncated_flows"] = int(
            frame.loc[frame["was_packet_truncated"], "flow_id"].nunique()
        )
    if "observed_packet_count" in frame:
        stats["observed_packet_count"] = {
            "min": int(frame["observed_packet_count"].min()),
            "median": float(frame["observed_packet_count"].median()),
            "max": int(frame["observed_packet_count"].max()),
        }
    if "split" in frame:
        stats["split_support"] = {
            "major_labels": support_by_split(frame, "major_label"),
            "source_labels": support_by_split(frame, "source_label"),
            "minor_labels": minor_support_by_split(frame),
        }
    for column in ["packet_count", "payload_byte_length", "packet_byte_length"]:
        if column in frame:
            stats[column] = {
                "min": int(frame[column].min()),
                "median": float(frame[column].median()),
                "max": int(frame[column].max()),
            }
    if metadata:
        stats.update(metadata)
    return stats


def split_run_stats(
    *,
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    split_method: str,
    max_per_class: int,
    stratify_column: str = "major_label",
    split_stratify_column: str = "source_label",
    group_column: str = "flow_id",
    time_column: str | None = None,
    block_size: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    metadata = {
        "split_method": split_method,
        "max_per_class": int(max_per_class),
        "block_size": None if block_size is None else int(block_size),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "stratify_column": stratify_column,
        "split_stratify_column": split_stratify_column,
        "group_column": group_column,
        "time_column": time_column,
        "sampling": {
            "input_rows": int(len(input_frame)),
            "input_flows": int(input_frame["flow_id"].nunique())
            if "flow_id" in input_frame
            else int(len(input_frame)),
            "output_rows": int(len(output_frame)),
            "output_flows": int(output_frame["flow_id"].nunique())
            if "flow_id" in output_frame
            else int(len(output_frame)),
        },
    }
    return processed_stats(output_frame, metadata=metadata)
