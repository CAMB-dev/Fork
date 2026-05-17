"""Audits for processed train/val/test splits."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import struct
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_LOW_SUPPORT_MAJOR_LABELS = ("infiltration", "web_attack", "botnet_malware")
MAX_AUDIT_WINDOW_CONTENT_TOKENS = 510
TOKEN_IDS = {
    "fwd": 5,
    "forward": 5,
    "client": 5,
    "bwd": 6,
    "backward": 6,
    "server": 6,
}
PKT_END_TOKEN_ID = 7
UNK_TOKEN_ID = 1
BYTE_TOKEN_OFFSET = 8
SUBMODE_SIGNATURE_VERSION = "coarse_packet_shape_v1"


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


def _as_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _bytes_sha1(value: object) -> str:
    return hashlib.sha1(_as_bytes(value)).hexdigest()


def _first_window_sha1(payload: object, packet_lengths: object, packet_directions: object) -> str:
    data = _as_bytes(payload)
    lengths = _as_list(packet_lengths)
    directions = _as_list(packet_directions)
    content_ids: list[int] = []
    offset = 0
    for direction, length_value in zip(directions, lengths, strict=False):
        if len(content_ids) >= MAX_AUDIT_WINDOW_CONTENT_TOKENS:
            break
        content_ids.append(TOKEN_IDS.get(str(direction).lower(), UNK_TOKEN_ID))
        length = int(length_value)
        take = min(length, MAX_AUDIT_WINDOW_CONTENT_TOKENS - len(content_ids))
        if take > 0:
            content_ids.extend(BYTE_TOKEN_OFFSET + value for value in data[offset : offset + take])
        offset += length
        if len(content_ids) >= MAX_AUDIT_WINDOW_CONTENT_TOKENS:
            break
        content_ids.append(PKT_END_TOKEN_ID)
    input_ids = [2, *content_ids, 3]
    if len(input_ids) < 512:
        input_ids.extend([0] * (512 - len(input_ids)))
    packed = b"".join(struct.pack("<H", int(value)) for value in input_ids[:512])
    return hashlib.sha1(packed).hexdigest()


def _hash_overlap(
    frames: dict[str, pd.DataFrame],
    column: str,
    sample_size: int = 10,
) -> dict[str, dict]:
    digest_column = f"_{column}_sha1"
    values_by_split = {
        split: set(frame[digest_column].dropna().map(str))
        if digest_column in frame
        else set(frame[column].map(_bytes_sha1))
        for split, frame in frames.items()
        if column in frame
    }
    overlap = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        values = sorted(values_by_split.get(left, set()) & values_by_split.get(right, set()))
        overlap[f"{left}_vs_{right}"] = {"count": len(values), "examples": values[:sample_size]}
    return overlap


def _first_window_hash_overlap(frames: dict[str, pd.DataFrame], sample_size: int = 10) -> dict[str, dict]:
    required = {"bytes", "packet_lengths", "packet_directions"}
    values_by_split = {}
    for split, frame in frames.items():
        if "_first_window_sha1" in frame:
            values_by_split[split] = set(frame["_first_window_sha1"].dropna().map(str))
            continue
        if not required <= set(frame.columns):
            continue
        values_by_split[split] = {
            _first_window_sha1(row.bytes, row.packet_lengths, row.packet_directions)
            for row in frame[list(required)].itertuples(index=False)
        }
    overlap = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        values = sorted(values_by_split.get(left, set()) & values_by_split.get(right, set()))
        overlap[f"{left}_vs_{right}"] = {"count": len(values), "examples": values[:sample_size]}
    return overlap


def _bucket_number(value: int, bins: tuple[int, ...]) -> str:
    value = int(value)
    previous = 0
    for bound in bins:
        if value <= bound:
            return f"{previous + 1}-{bound}"
        previous = bound
    return f">{bins[-1]}"


def _bucket_payload_length(value: object) -> str:
    length = int(value) if value is not None else 0
    if length <= 0:
        return "0"
    return _bucket_number(length, (31, 127, 511, 2047, 8191))


def _bucket_packet_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return str(value)
    return _bucket_number(value, (4, 8, 16, 32, 64))


def _packet_length_bins(lengths: object, limit: int = 8) -> tuple[str, ...]:
    return tuple(_bucket_number(int(value), (63, 127, 511, 1500)) for value in _as_list(lengths)[:limit])


def _direction_pattern(directions: object, limit: int = 8) -> str:
    tokens = []
    for value in _as_list(directions)[:limit]:
        text = str(value).lower()
        if text in {"fwd", "forward", "client"}:
            tokens.append("f")
        elif text in {"bwd", "backward", "server"}:
            tokens.append("b")
        else:
            tokens.append("?")
    return "".join(tokens)


def _submode_signature(row: object) -> tuple:
    lengths = _as_list(getattr(row, "packet_lengths", None))
    return (
        str(getattr(row, "connection_type", "")),
        _bucket_packet_count(len(lengths)),
        _bucket_payload_length(getattr(row, "payload_byte_length", 0)),
        _direction_pattern(getattr(row, "packet_directions", None)),
        _packet_length_bins(lengths),
    )


def _submode_signature_id(signature: tuple) -> str:
    text = repr(signature).encode("utf-8")
    return hashlib.sha1(text).hexdigest()[:16]


def _submode_signature_payload(signature: tuple) -> dict:
    return {
        "connection_type": str(signature[0]),
        "packet_count_bucket": str(signature[1]),
        "payload_byte_length_bucket": str(signature[2]),
        "direction_pattern": str(signature[3]),
        "packet_length_bins": list(signature[4]),
    }


def label_submode_train_coverage(
    frames: dict[str, pd.DataFrame],
    *,
    label_columns: Iterable[str] = ("major_label", "source_label"),
    min_train_support: int = 100,
    min_eval_support: int = 100,
    max_unseen_ratio: float = 0.20,
    min_ambiguous_unseen_rows: int = 50,
    sample_size: int = 10,
) -> dict:
    """Detect attack submodes that evaluation sees but training does not.

    The signature is intentionally coarse and model-visible: connection type,
    packet-count bucket, payload-length bucket, direction pattern, and packet
    length bins. A split is risky when an attack label's eval rows mostly fall
    into signatures absent from train for that label, while those signatures are
    present in train under another label. That pattern means the model was
    explicitly trained to associate the visible shape with a different class.
    """

    required = {"packet_lengths", "packet_directions", "connection_type", "payload_byte_length"}
    label_columns = tuple(label_columns)
    if not all(required <= set(frame.columns) for frame in frames.values()):
        return {
            "signature_version": SUBMODE_SIGNATURE_VERSION,
            "available": False,
            "blocking_items": [],
            "warnings": [],
            "reason": "missing packet shape columns",
        }

    signature_examples: dict[str, dict] = {}
    count_by_split_label_signature: Counter[tuple[str, str, str, str]] = Counter()
    train_signatures_by_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    train_labels_by_signature: dict[str, set[tuple[str, str]]] = defaultdict(set)
    support_by_split_label: Counter[tuple[str, str, str]] = Counter()

    usable_label_columns = [
        column for column in label_columns if all(column in frame.columns for frame in frames.values())
    ]
    for split, frame in frames.items():
        columns = [*usable_label_columns, "packet_lengths", "packet_directions", "connection_type", "payload_byte_length"]
        for row in frame[columns].itertuples(index=False):
            signature = _submode_signature(row)
            signature_id = _submode_signature_id(signature)
            signature_examples.setdefault(signature_id, _submode_signature_payload(signature))
            for label_column in usable_label_columns:
                label = str(getattr(row, label_column))
                if split == "train":
                    train_labels_by_signature[signature_id].add((label_column, label))
                if label.lower() == "benign":
                    continue
                key = (label_column, label)
                support_by_split_label[(split, label_column, label)] += 1
                count_by_split_label_signature[(split, label_column, label, signature_id)] += 1
                if split == "train":
                    train_signatures_by_label[key].add(signature_id)

    items = []
    warnings = []
    for split in ("val", "test"):
        labels = sorted(
            (label_column, label)
            for split_name, label_column, label in support_by_split_label
            if split_name == split
        )
        for label_column, label in labels:
            eval_support = int(support_by_split_label[(split, label_column, label)])
            train_support = int(support_by_split_label[("train", label_column, label)])
            if eval_support < min_eval_support or train_support < min_train_support:
                continue
            train_signatures = train_signatures_by_label.get((label_column, label), set())
            unseen_rows = 0
            ambiguous_unseen_rows = 0
            unseen_signatures = 0
            examples = []
            for (split_name, column_name, label_name, signature_id), rows in count_by_split_label_signature.items():
                if split_name != split or column_name != label_column or label_name != label:
                    continue
                if signature_id in train_signatures:
                    continue
                unseen_rows += int(rows)
                unseen_signatures += 1
                other_train_labels = sorted(
                    f"{column}:{value}"
                    for column, value in train_labels_by_signature.get(signature_id, set())
                    if not (column == label_column and value == label)
                )
                if other_train_labels:
                    ambiguous_unseen_rows += int(rows)
                    if len(examples) < sample_size:
                        examples.append(
                            {
                                "signature_id": signature_id,
                                "rows": int(rows),
                                "train_labels_for_same_shape": other_train_labels[:8],
                                "signature": signature_examples.get(signature_id, {}),
                            }
                        )
            if not unseen_rows:
                continue
            unseen_ratio = unseen_rows / eval_support
            ambiguous_ratio = ambiguous_unseen_rows / eval_support
            item = {
                "split": split,
                "label_column": label_column,
                "label": label,
                "train_support": train_support,
                "eval_support": eval_support,
                "unseen_rows": int(unseen_rows),
                "unseen_ratio": unseen_ratio,
                "unseen_signature_count": int(unseen_signatures),
                "ambiguous_unseen_rows": int(ambiguous_unseen_rows),
                "ambiguous_unseen_ratio": ambiguous_ratio,
                "examples": examples,
            }
            items.append(item)
            if unseen_ratio > max_unseen_ratio and ambiguous_unseen_rows >= min_ambiguous_unseen_rows:
                warnings.append(
                    "label submode train coverage is weak for "
                    f"{label_column}={label} in {split}: "
                    f"{unseen_rows}/{eval_support} eval rows use train-unseen "
                    f"visible shapes, {ambiguous_unseen_rows} of them appear "
                    "in train under other labels"
                )

    return {
        "signature_version": SUBMODE_SIGNATURE_VERSION,
        "available": True,
        "min_train_support": min_train_support,
        "min_eval_support": min_eval_support,
        "max_unseen_ratio": max_unseen_ratio,
        "min_ambiguous_unseen_rows": min_ambiguous_unseen_rows,
        "items": items,
        "blocking_items": [
            item
            for item in items
            if item["unseen_ratio"] > max_unseen_ratio
            and item["ambiguous_unseen_rows"] >= min_ambiguous_unseen_rows
        ],
        "warnings": warnings,
    }


def _canonical_tuple_row(row: tuple) -> tuple[str, str, str]:
    protocol, endpoint_a, endpoint_b = row
    left, right = sorted([str(endpoint_a), str(endpoint_b)])
    return str(protocol), left, right


def nearby_five_tuple_split_leakage(
    frame: pd.DataFrame,
    threshold_seconds: float = 120.0,
    sample_size: int = 10,
) -> dict:
    required = {"flow_id", "split", "protocol", "endpoint_a", "endpoint_b", "start_time"}
    if not required <= set(frame.columns):
        return {"threshold_seconds": threshold_seconds, "count": 0, "examples": []}

    work = frame[list(required)].dropna(subset=["start_time"]).drop_duplicates("flow_id").copy()
    endpoint_a = work["endpoint_a"].astype(str)
    endpoint_b = work["endpoint_b"].astype(str)
    left = endpoint_a.where(endpoint_a <= endpoint_b, endpoint_b)
    right = endpoint_b.where(endpoint_a <= endpoint_b, endpoint_a)
    work["_tuple"] = work["protocol"].astype(str) + "|" + left + "|" + right
    work = work.sort_values(["_tuple", "start_time", "flow_id"], kind="mergesort")

    times = pd.to_numeric(work["start_time"], errors="coerce")
    same_tuple = work["_tuple"].eq(work["_tuple"].shift())
    nearby = same_tuple & times.diff().le(float(threshold_seconds))
    split_changed = work["split"].astype(str).ne(work["split"].shift().astype(str))
    leaked = work.loc[nearby & split_changed]
    count = int(len(leaked))
    examples = []
    if count:
        previous_rows = work.shift().loc[leaked.index]
        deltas = times.diff().loc[leaked.index]
        for (_, row), (_, previous), delta in zip(
            leaked.head(sample_size).iterrows(),
            previous_rows.head(sample_size).iterrows(),
            deltas.head(sample_size),
            strict=True,
        ):
            examples.append(
                {
                    "five_tuple": str(row["_tuple"]).split("|", 2),
                    "left_flow_id": str(previous.flow_id),
                    "left_split": str(previous.split),
                    "left_start_time": float(previous.start_time),
                    "right_flow_id": str(row.flow_id),
                    "right_split": str(row.split),
                    "right_start_time": float(row.start_time),
                    "delta_seconds": float(delta),
                }
            )
    return {
        "threshold_seconds": threshold_seconds,
        "count": count,
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
        "observed_packet_count",
        "was_packet_truncated",
        "payload_byte_length",
        "bytes",
        "_bytes_sha1",
        "_first_window_sha1",
        "packet_lengths",
        "packet_directions",
        "connection_type",
        "label_match_mode",
        "label_time_delta_seconds",
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
        "connection_types": counts(frame, "connection_type"),
        "label_match_modes": counts(frame, "label_match_mode"),
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
        "packet_truncation": {
            "rows": int(frame["was_packet_truncated"].sum())
            if "was_packet_truncated" in frame
            else 0,
            "flows": int(frame.loc[frame["was_packet_truncated"], "flow_id"].nunique())
            if {"was_packet_truncated", "flow_id"} <= set(frame.columns)
            else 0,
            "max_observed_packet_count": int(frame["observed_packet_count"].max())
            if "observed_packet_count" in frame and len(frame)
            else None,
        },
    }


def audit_split_frames(
    frames: dict[str, pd.DataFrame],
    *,
    nearby_threshold_seconds: float = 120.0,
    low_support_min: int = 2,
    low_support_labels: Iterable[str] = DEFAULT_LOW_SUPPORT_MAJOR_LABELS,
    ignore_source_file_overlap_for_eligibility: bool = False,
    submode_min_train_support: int = 100,
    submode_min_eval_support: int = 100,
    submode_max_unseen_ratio: float = 0.20,
    submode_min_ambiguous_unseen_rows: int = 50,
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
    major_totals = counts(all_frame, "major_label")
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
    low_support = low_support_audit(
        all_frame,
        labels=low_support_labels,
        min_support=low_support_min,
    )
    bytes_overlap = _hash_overlap(frames, "bytes")
    first_window_overlap = _first_window_hash_overlap(frames)
    submode_coverage = label_submode_train_coverage(
        frames,
        min_train_support=submode_min_train_support,
        min_eval_support=submode_min_eval_support,
        max_unseen_ratio=submode_max_unseen_ratio,
        min_ambiguous_unseen_rows=submode_min_ambiguous_unseen_rows,
    )

    warnings = []
    for pair, item in overlap.items():
        if item["count"] > 0:
            warnings.append(f"source_file overlap detected in {pair}: {item['count']}")
    if len(all_major_labels) < 2:
        label_text = ", ".join(all_major_labels) if all_major_labels else "none"
        warnings.append(f"fewer than 2 major labels across all splits: {label_text}")
    for split, missing in missing_major_by_split.items():
        if missing:
            blocking_missing = [
                label for label in missing if int(major_totals.get(label, 0)) >= 3
            ]
            low_total_missing = [
                label for label in missing if int(major_totals.get(label, 0)) < 3
            ]
            if blocking_missing:
                warnings.append(
                    f"{split} is missing major labels: {', '.join(blocking_missing)}"
                )
            for label in low_total_missing:
                warnings.append(
                    f"low-support major label {label} is absent from {split}; "
                    f"total support is {int(major_totals.get(label, 0))}"
                )
    if flow_leakage["count"] > 0:
        warnings.append(f"flow_id leakage detected across splits: {flow_leakage['count']}")
    if nearby_leakage["count"] > 0:
        warnings.append(
            "nearby five-tuple split leakage detected within "
            f"{nearby_threshold_seconds:g}s: {nearby_leakage['count']}"
        )
    for pair, item in bytes_overlap.items():
        if item["count"] > 0:
            warnings.append(f"bytes overlap detected in {pair}: {item['count']}")
    for pair, item in first_window_overlap.items():
        if item["count"] > 0:
            warnings.append(f"first-window token overlap detected in {pair}: {item['count']}")
    warnings.extend(low_support["warnings"])
    warnings.extend(submode_coverage["warnings"])

    blocking_warnings = [
        warning
        for warning in warnings
        if not (
            ignore_source_file_overlap_for_eligibility
            and warning.startswith("source_file overlap detected")
        )
        and not warning.startswith("low-support major label ")
    ]

    return {
        "splits": {split: split_summary(frame) for split, frame in frames.items()},
        "all_major_labels": all_major_labels,
        "source_file_overlap": overlap,
        "bytes_hash_overlap": bytes_overlap,
        "first_window_hash_overlap": first_window_overlap,
        "missing_major_by_split": missing_major_by_split,
        "flow_id_leakage": flow_leakage,
        "nearby_five_tuple_split_leakage": nearby_leakage,
        "label_submode_train_coverage": submode_coverage,
        "support_by_split": {
            "major_labels": support_by_split(all_frame, "major_label"),
            "source_labels": support_by_split(all_frame, "source_label"),
            "minor_labels": minor_support_by_split(all_frame),
            "connection_types": support_by_split(all_frame, "connection_type"),
            "label_match_modes": support_by_split(all_frame, "label_match_mode"),
        },
        "low_support": low_support,
        "formal_eligible": len(blocking_warnings) == 0,
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def audit_split_paths(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    *,
    nearby_threshold_seconds: float = 120.0,
    low_support_min: int = 2,
    low_support_labels: Iterable[str] = DEFAULT_LOW_SUPPORT_MAJOR_LABELS,
    ignore_source_file_overlap_for_eligibility: bool = False,
    submode_min_train_support: int = 100,
    submode_min_eval_support: int = 100,
    submode_max_unseen_ratio: float = 0.20,
    submode_min_ambiguous_unseen_rows: int = 50,
) -> dict:
    return audit_split_frames(
        {
            "train": read_split(train_path),
            "val": read_split(val_path),
            "test": read_split(test_path),
        },
        nearby_threshold_seconds=nearby_threshold_seconds,
        low_support_min=low_support_min,
        low_support_labels=low_support_labels,
        ignore_source_file_overlap_for_eligibility=ignore_source_file_overlap_for_eligibility,
        submode_min_train_support=submode_min_train_support,
        submode_min_eval_support=submode_min_eval_support,
        submode_max_unseen_ratio=submode_max_unseen_ratio,
        submode_min_ambiguous_unseen_rows=submode_min_ambiguous_unseen_rows,
    )
