"""Audits for processed train/val/test splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_LOW_SUPPORT_MAJOR_LABELS = ("infiltration", "web_attack", "botnet_malware")


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in Counter(frame[column].dropna()).items()}


def flatten_minor_labels(values: pd.Series) -> list[str]:
    labels: list[str] = []
    for value in values:
        if value is None:
            continue
        labels.extend(str(label) for label in value)
    return labels


def support_by_split(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    if "split" not in frame or column not in frame:
        return {}
    grouped = frame.groupby(["split", column], dropna=False).size().reset_index(name="rows")
    output: dict[str, dict[str, int]] = {}
    for row in grouped.itertuples(index=False):
        split = str(getattr(row, "split"))
        label = str(getattr(row, column))
        output.setdefault(split, {})[label] = int(row.rows)
    return output


def minor_support_by_split(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    if "split" not in frame or "minor_labels" not in frame:
        return {}
    rows = []
    for split, labels in frame[["split", "minor_labels"]].itertuples(index=False, name=None):
        if labels is None:
            continue
        for label in labels:
            rows.append({"split": split, "minor_label": str(label)})
    if not rows:
        return {}
    return support_by_split(pd.DataFrame(rows), "minor_label")


def flow_id_leakage(frame: pd.DataFrame, sample_size: int = 10) -> dict:
    if not {"flow_id", "split"} <= set(frame.columns):
        return {"count": 0, "examples": []}
    grouped = frame.groupby("flow_id")["split"].agg(lambda values: sorted(set(map(str, values))))
    leaked = grouped[grouped.map(len) > 1]
    examples = [
        {"flow_id": str(flow_id), "splits": splits}
        for flow_id, splits in leaked.head(sample_size).items()
    ]
    return {"count": int(len(leaked)), "examples": examples}


def _canonical_tuple_row(row: tuple) -> tuple[str, str, str]:
    protocol, endpoint_a, endpoint_b = row
    left, right = sorted([str(endpoint_a), str(endpoint_b)])
    return str(protocol), left, right


def nearby_five_tuple_split_leakage(
    frame: pd.DataFrame,
    threshold_seconds: float = 60.0,
    sample_size: int = 10,
) -> dict:
    required = {"flow_id", "split", "protocol", "endpoint_a", "endpoint_b", "start_time"}
    if not required <= set(frame.columns):
        return {"threshold_seconds": threshold_seconds, "count": 0, "examples": []}

    work = frame[list(required)].dropna(subset=["start_time"]).drop_duplicates("flow_id").copy()
    work["_tuple"] = [
        _canonical_tuple_row(row)
        for row in work[["protocol", "endpoint_a", "endpoint_b"]].itertuples(
            index=False,
            name=None,
        )
    ]
    work = work.sort_values(["_tuple", "start_time", "flow_id"], kind="mergesort")

    count = 0
    examples = []
    for flow_tuple, group in work.groupby("_tuple", sort=False):
        previous = None
        for row in group.itertuples(index=False):
            if previous is None:
                previous = row
                continue
            delta = float(row.start_time) - float(previous.start_time)
            if delta <= threshold_seconds and str(row.split) != str(previous.split):
                count += 1
                if len(examples) < sample_size:
                    examples.append(
                        {
                            "five_tuple": list(flow_tuple),
                            "left_flow_id": str(previous.flow_id),
                            "left_split": str(previous.split),
                            "left_start_time": float(previous.start_time),
                            "right_flow_id": str(row.flow_id),
                            "right_split": str(row.split),
                            "right_start_time": float(row.start_time),
                            "delta_seconds": delta,
                        }
                    )
            previous = row
    return {
        "threshold_seconds": threshold_seconds,
        "count": int(count),
        "examples": examples,
    }


def low_support_audit(
    frame: pd.DataFrame,
    labels: Iterable[str] = DEFAULT_LOW_SUPPORT_MAJOR_LABELS,
    min_support: int = 2,
) -> dict:
    support = support_by_split(frame, "major_label")
    labels = tuple(labels)
    by_label = {}
    warnings = []
    for label in labels:
        counts_by_split = {
            split: int(support.get(split, {}).get(label, 0))
            for split in ["train", "val", "test"]
        }
        by_label[label] = counts_by_split
        for split in ["val", "test"]:
            if counts_by_split[split] < min_support:
                warnings.append(
                    f"{label} has support {counts_by_split[split]} in {split}, "
                    f"below {min_support}"
                )
    return {"min_support": min_support, "major_labels": by_label, "warnings": warnings}


def read_split(path: Path) -> pd.DataFrame:
    columns = [
        "flow_id",
        "source_file",
        "source_label",
        "major_label",
        "minor_labels",
        "view",
        "protocol",
        "endpoint_a",
        "endpoint_b",
        "start_time",
        "packet_count",
        "payload_byte_length",
    ]
    available = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[column for column in columns if column in available])


def split_summary(frame: pd.DataFrame) -> dict:
    source_files = set(str(value) for value in frame.get("source_file", pd.Series(dtype=str)).dropna())
    return {
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()) if "flow_id" in frame else int(len(frame)),
        "major_labels": counts(frame, "major_label"),
        "source_labels": counts(frame, "source_label"),
        "views": counts(frame, "view"),
        "source_file_count": len(source_files),
        "packet_count": {
            "min": int(frame["packet_count"].min()) if "packet_count" in frame and len(frame) else None,
            "median": float(frame["packet_count"].median())
            if "packet_count" in frame and len(frame)
            else None,
            "max": int(frame["packet_count"].max()) if "packet_count" in frame and len(frame) else None,
        },
        "payload_byte_length": {
            "min": int(frame["payload_byte_length"].min())
            if "payload_byte_length" in frame and len(frame)
            else None,
            "median": float(frame["payload_byte_length"].median())
            if "payload_byte_length" in frame and len(frame)
            else None,
            "max": int(frame["payload_byte_length"].max())
            if "payload_byte_length" in frame and len(frame)
            else None,
        },
    }


def audit_split_frames(
    frames: dict[str, pd.DataFrame],
    *,
    nearby_threshold_seconds: float = 60.0,
    low_support_min: int = 2,
) -> dict:
    source_files = {
        split: set(str(value) for value in frame.get("source_file", pd.Series(dtype=str)).dropna())
        for split, frame in frames.items()
    }
    all_frame = pd.concat(
        [frame.assign(split=split) for split, frame in frames.items()],
        ignore_index=True,
    )
    all_major_labels = sorted(
        str(label)
        for label in all_frame.get("major_label", pd.Series(dtype=str)).dropna().unique()
    )
    overlap = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        values = sorted(source_files.get(left, set()) & source_files.get(right, set()))
        overlap[f"{left}_vs_{right}"] = {
            "count": len(values),
            "examples": values[:10],
        }

    missing_major_by_split = {}
    for split, frame in frames.items():
        present = set(str(label) for label in frame.get("major_label", pd.Series(dtype=str)).dropna().unique())
        missing_major_by_split[split] = [label for label in all_major_labels if label not in present]

    flow_leakage = flow_id_leakage(all_frame)
    nearby_leakage = nearby_five_tuple_split_leakage(
        all_frame,
        threshold_seconds=nearby_threshold_seconds,
    )
    low_support = low_support_audit(all_frame, min_support=low_support_min)

    warnings = []
    for pair, item in overlap.items():
        if item["count"] > 0:
            warnings.append(f"source_file overlap detected in {pair}: {item['count']}")
    if len(all_major_labels) < 2:
        label_text = ", ".join(all_major_labels) if all_major_labels else "none"
        warnings.append(f"fewer than 2 major labels across all splits: {label_text}")
    for split, missing in missing_major_by_split.items():
        if missing:
            warnings.append(f"{split} is missing major labels: {', '.join(missing)}")
    if flow_leakage["count"] > 0:
        warnings.append(f"flow_id leakage detected across splits: {flow_leakage['count']}")
    if nearby_leakage["count"] > 0:
        warnings.append(
            "nearby five-tuple split leakage detected within "
            f"{nearby_threshold_seconds:g}s: {nearby_leakage['count']}"
        )
    warnings.extend(low_support["warnings"])

    return {
        "splits": {split: split_summary(frame) for split, frame in frames.items()},
        "all_major_labels": all_major_labels,
        "source_file_overlap": overlap,
        "missing_major_by_split": missing_major_by_split,
        "flow_id_leakage": flow_leakage,
        "nearby_five_tuple_split_leakage": nearby_leakage,
        "support_by_split": {
            "major_labels": support_by_split(all_frame, "major_label"),
            "source_labels": support_by_split(all_frame, "source_label"),
            "minor_labels": minor_support_by_split(all_frame),
        },
        "low_support": low_support,
        "warnings": warnings,
    }


def audit_split_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    *,
    nearby_threshold_seconds: float = 60.0,
    low_support_min: int = 2,
) -> dict:
    return audit_split_frames(
        {
            "train": read_split(train_path),
            "val": read_split(val_path),
            "test": read_split(test_path),
        },
        nearby_threshold_seconds=nearby_threshold_seconds,
        low_support_min=low_support_min,
    )
