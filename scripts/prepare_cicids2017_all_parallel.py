"""Prepare CICIDS2017 working-hour PCAPs in parallel.

The script builds one processed Parquet shard per official labelled-flow CSV,
then merges all shards and writes a stratified train/val/test split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_bert.config import write_json
from traffic_bert.data.cicids_audit import (
    build_cicids_coverage_audit,
    write_cicids_coverage_artifacts,
)
from traffic_bert.data.build import (
    BuildConfig,
    build_processed_dataset,
    build_processed_dataset_from_flows,
)
from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.schema import InputView
from traffic_bert.data.split import (
    assign_time_block_split,
    assign_time_ordered_split,
    assign_stratified_hash_split,
    processed_stats,
    split_run_stats,
    stable_bucket,
    stratified_sample,
)
from traffic_bert.data.split_audit import (
    _bytes_sha1,
    _first_window_sha1,
    _submode_signature,
    _submode_signature_id,
)
from traffic_bert.data.validate import validation_summary


@dataclass(frozen=True)
class CicidsJob:
    slug: str
    pcap_name: str
    csv_contains: tuple[str, ...]
    attack_labels: tuple[str, ...]


class AttackCoverageError(RuntimeError):
    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


JOBS = (
    CicidsJob("monday_benign", "Monday-WorkingHours.pcap", ("Monday",), ()),
    CicidsJob(
        "tuesday_patator",
        "Tuesday-WorkingHours.pcap",
        ("Tuesday",),
        ("FTP-Patator", "SSH-Patator"),
    ),
    CicidsJob(
        "wednesday_dos",
        "Wednesday-workingHours.pcap",
        ("Wednesday",),
        ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"),
    ),
    CicidsJob(
        "thursday_web",
        "Thursday-WorkingHours.pcap",
        ("Thursday", "WebAttacks"),
        (
            "Web Attack \x96 Brute Force",
            "Web Attack \x96 XSS",
            "Web Attack \x96 Sql Injection",
        ),
    ),
    CicidsJob(
        "thursday_infiltration",
        "Thursday-WorkingHours.pcap",
        ("Thursday", "Infilteration"),
        ("Infiltration",),
    ),
    CicidsJob("friday_bot", "Friday-WorkingHours.pcap", ("Friday", "Morning"), ("Bot",)),
    CicidsJob(
        "friday_portscan",
        "Friday-WorkingHours.pcap",
        ("Friday", "PortScan"),
        ("PortScan",),
    ),
    CicidsJob("friday_ddos", "Friday-WorkingHours.pcap", ("Friday", "DDos"), ("DDoS",)),
)

TIMESTAMP_POLICY_VERSION = 2
FULL_DAY_PM_CUTOFF_HOUR = 8
DEFAULT_CIC_LABEL_MAX_TIME_DELTA_SECONDS = 15 * 60
FORMAL_GROUP_CACHE_VERSION = "visible_hash_nearby_v1"


def _log_event(event: str, **payload: object) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False),
        flush=True,
    )


def _validate_required_pcaps(raw_dir: Path) -> None:
    missing = []
    for pcap_name in dict.fromkeys(job.pcap_name for job in JOBS):
        path = raw_dir / "pcaps" / pcap_name
        if not path.exists():
            missing.append(pcap_name)
            continue
        if path.stat().st_size <= 0:
            missing.append(f"{pcap_name} (empty)")
    if missing:
        joined = ", ".join(dict.fromkeys(missing))
        raise FileNotFoundError(
            "CICIDS2017 full preprocessing requires all Monday-Friday working-hour "
            f"PCAPs before any formal split is written. Missing: {joined}. "
            "Run DATASET=cicids2017-all bash scripts/download_datasets.sh."
        )


def _find_csv_name(label_zip: Path, filename_contains: tuple[str, ...]) -> str:
    needles = tuple(item.lower() for item in filename_contains)
    with zipfile.ZipFile(label_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and all(needle in Path(name).name.lower() for needle in needles)
        ]
    if not matches:
        raise FileNotFoundError(f"no CSV containing {filename_contains!r} in {label_zip}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous CSVs for {filename_contains!r}: {matches}")
    return matches[0]


def _read_label_csv(label_zip: Path, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(label_zip) as archive:
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, encoding="latin1", low_memory=False)


def _column(frame: pd.DataFrame, name: str) -> str:
    for column in frame.columns:
        if column.strip().lower() == name:
            return column
    raise ValueError(f"CSV has no {name!r} column")


def _unix_seconds(value: pd.Timestamp) -> float:
    if value.tzinfo is None:
        value = value.tz_localize(timezone.utc)
    return float(value.timestamp())


def _parse_cicids_timestamps(values: pd.Series, csv_name: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", dayfirst=True)
    name = Path(csv_name).name.lower()
    valid = timestamps.notna()

    if "afternoon" in name:
        morning_like = valid & (timestamps.dt.hour < 12)
        timestamps.loc[morning_like] = timestamps.loc[morning_like] + pd.Timedelta(hours=12)
        return timestamps

    # The full-day CICIDS CSVs are exported without AM/PM, while their PCAPs
    # cover roughly 12:00-20:00 UTC. Hours before this cutoff are afternoon
    # traffic written as 1-7 instead of 13-19.
    if "workinghours.pcap_iscx.csv" in name:
        ambiguous_pm = valid & (timestamps.dt.hour < FULL_DAY_PM_CUTOFF_HOUR)
        timestamps.loc[ambiguous_pm] = timestamps.loc[ambiguous_pm] + pd.Timedelta(hours=12)
    return timestamps


def _resolve_pcap(raw_dir: Path, pcap_name: str) -> Path:
    direct = raw_dir / "pcaps" / pcap_name
    if direct.exists():
        return direct
    candidates = sorted((raw_dir / "pcaps").glob(f"*{pcap_name.split('-')[0]}*.pcap"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"missing PCAP for {pcap_name}; run DATASET=cicids2017-all "
        "bash scripts/download_datasets.sh"
    )


def _extract_label_file(
    *,
    raw_dir: Path,
    label_zip: Path,
    job: CicidsJob,
    padding_minutes: int,
    csv_time_offset_hours: float,
    window_scope: str,
) -> dict:
    csv_name = _find_csv_name(label_zip, job.csv_contains)
    frame = _read_label_csv(label_zip, csv_name)
    label_col = _column(frame, "label")
    ts_col = _column(frame, "timestamp")

    timestamps = _parse_cicids_timestamps(frame[ts_col], csv_name)
    original_min_time = timestamps.dropna().min()
    original_max_time = timestamps.dropna().max()
    if csv_time_offset_hours:
        timestamps = timestamps + pd.Timedelta(hours=csv_time_offset_hours)
    frame[ts_col] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    labels_dir = raw_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_path = labels_dir / f"cicids2017_{job.slug}.csv"
    frame.to_csv(output_path, index=False)
    if window_scope == "pcap":
        selected_times = pd.Series([], dtype="datetime64[ns]")
        min_time = pd.NaT
        max_time = pd.NaT
    elif window_scope == "attack" and job.attack_labels:
        attack_mask = frame[label_col].isin(job.attack_labels)
        selected_times = timestamps[attack_mask & timestamps.notna()]
    else:
        selected_times = timestamps[timestamps.notna()]

    if window_scope != "pcap" and selected_times.empty:
        min_time = timestamps.dropna().min()
        max_time = timestamps.dropna().max()
    elif window_scope != "pcap":
        min_time = selected_times.min() - pd.Timedelta(minutes=padding_minutes)
        max_time = selected_times.max() + pd.Timedelta(minutes=padding_minutes)

    summary = {
        "slug": job.slug,
        "pcap_name": job.pcap_name,
        "csv_name": csv_name,
        "output_path": str(output_path),
        "rows": int(len(frame)),
        "label_counts": frame[label_col].value_counts().to_dict(),
        "attack_labels": list(job.attack_labels),
        "csv_time_offset_hours": csv_time_offset_hours,
        "window_scope": window_scope,
        "original_timestamp_start": None if pd.isna(original_min_time) else str(original_min_time),
        "original_timestamp_end": None if pd.isna(original_max_time) else str(original_max_time),
        "window_start": None if pd.isna(min_time) else str(min_time),
        "window_end": None if pd.isna(max_time) else str(max_time),
        "window_start_unix": None if pd.isna(min_time) else _unix_seconds(min_time),
        "window_end_unix": None if pd.isna(max_time) else _unix_seconds(max_time),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _config_from_payload(payload: dict) -> BuildConfig:
    return BuildConfig(
        input_path=Path(payload["pcap_path"]),
        output_path=Path(payload["output_path"]),
        label_map_path=Path(payload["label_map"]),
        source_dataset="cicids2017",
        split="train",
        label_source="cic_csv",
        label_csv_path=Path(payload["label_csv"]),
        drop_unmatched_labels=True,
        views=tuple(InputView(item) for item in payload["views"]),
        keep_empty_payload=bool(payload["keep_empty_payload"]),
        max_packets_per_flow=int(payload["max_packets_per_flow"]),
        max_packets_to_read=payload["max_packets_to_read"],
        min_packet_time=payload["min_packet_time"],
        max_packet_time=payload["max_packet_time"],
        flow_timeout_seconds=payload["flow_timeout_seconds"],
        close_on_tcp_flags=payload.get("close_on_tcp_flags", False),
        cic_label_max_time_delta_seconds=payload["cic_label_max_time_delta_seconds"],
    )


def _build_metadata(config: BuildConfig, payload: dict, stats: dict) -> dict:
    return {
        **stats,
        "label_csv": str(config.label_csv_path),
        "min_packet_time": config.min_packet_time,
        "max_packet_time": config.max_packet_time,
        "max_packets_to_read": config.max_packets_to_read,
        "max_packets_per_flow": config.max_packets_per_flow,
        "flow_timeout_seconds": config.flow_timeout_seconds,
        "cic_label_max_time_delta_seconds": config.cic_label_max_time_delta_seconds,
        "csv_time_offset_hours": payload["csv_time_offset_hours"],
        "cicids_timestamp_policy_version": payload["cicids_timestamp_policy_version"],
        "window_scope": payload["window_scope"],
        "views": list(payload["views"]),
        "keep_empty_payload": payload["keep_empty_payload"],
        "connection_type_version": 1,
        "tcp_close_policy_version": 3,
    }


def _write_build_metadata(config: BuildConfig, payload: dict, stats: dict) -> None:
    write_json(config.output_path.with_suffix(".build.json"), _build_metadata(config, payload, stats))


def _build_one(payload: dict) -> dict:
    config = _config_from_payload(payload)
    stats = build_processed_dataset(config)
    _write_build_metadata(config, payload, stats)
    return {"output_path": str(config.output_path), "stats": stats}


def _extract_key(payload: dict) -> tuple:
    return (
        payload["pcap_path"],
        payload["max_packets_per_flow"],
        payload["max_packets_to_read"],
        payload["min_packet_time"],
        payload["max_packet_time"],
        payload["flow_timeout_seconds"],
        payload.get("close_on_tcp_flags", False),
    )


def _build_group(payloads: list[dict]) -> list[dict]:
    first = payloads[0]
    pcap_path = Path(first["pcap_path"])
    shard_names = ", ".join(Path(payload["output_path"]).name for payload in payloads)
    started = time.monotonic()
    _log_event(
        "cicids_pcap_scan_start",
        pid=os.getpid(),
        pcap_path=str(pcap_path),
        shards=shard_names,
        max_packets_per_flow=first["max_packets_per_flow"],
        flow_timeout_seconds=first["flow_timeout_seconds"],
        min_packet_time=first["min_packet_time"],
        max_packet_time=first["max_packet_time"],
    )
    extractor = PcapFlowExtractor(
        max_packets_per_flow=first["max_packets_per_flow"],
        max_packets_to_read=first["max_packets_to_read"],
        min_packet_time=first["min_packet_time"],
        max_packet_time=first["max_packet_time"],
        flow_timeout_seconds=first["flow_timeout_seconds"],
        close_on_tcp_flags=first.get("close_on_tcp_flags", False),
    )
    flows_by_file = {pcap_path: extractor.extract(pcap_path)}
    _log_event(
        "cicids_pcap_scan_done",
        pid=os.getpid(),
        pcap_path=str(pcap_path),
        flows=len(flows_by_file[pcap_path]),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )

    results: list[dict] = []
    for payload in payloads:
        config = _config_from_payload(payload)
        shard_started = time.monotonic()
        _log_event(
            "cicids_shard_build_start",
            pid=os.getpid(),
            output_path=str(config.output_path),
        )
        stats = build_processed_dataset_from_flows(config, flows_by_file)
        _write_build_metadata(config, payload, stats)
        _log_event(
            "cicids_shard_build_done",
            pid=os.getpid(),
            output_path=str(config.output_path),
            rows=stats.get("rows"),
            flows=stats.get("flows"),
            elapsed_seconds=round(time.monotonic() - shard_started, 3),
        )
        results.append({"output_path": str(config.output_path), "stats": stats})
    return results


def _is_fresh_shard(output_path: Path, payload: dict) -> bool:
    if not output_path.exists():
        return False
    build_path = output_path.with_suffix(".build.json")
    if not build_path.exists():
        return False
    try:
        with open(build_path, encoding="utf-8") as handle:
            build = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "max_packets_per_flow": payload["max_packets_per_flow"],
        "max_packets_to_read": payload["max_packets_to_read"],
        "flow_timeout_seconds": payload["flow_timeout_seconds"],
        "min_packet_time": payload["min_packet_time"],
        "max_packet_time": payload["max_packet_time"],
        "csv_time_offset_hours": payload["csv_time_offset_hours"],
        "cicids_timestamp_policy_version": payload["cicids_timestamp_policy_version"],
        "cic_label_max_time_delta_seconds": payload["cic_label_max_time_delta_seconds"],
        "window_scope": payload["window_scope"],
        "views": payload.get("views"),
        "keep_empty_payload": payload.get("keep_empty_payload"),
        "connection_type_version": 1,
        "tcp_close_policy_version": 3,
    }
    return all(build.get(key) == value for key, value in expected.items())


def _merge_outputs(paths: list[Path], output_path: Path) -> dict:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("no processed CICIDS2017 shard outputs were created")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(
        subset=["flow_id", "view", "source_label", "major_label"]
    ).sort_values(["start_time", "flow_id"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), output_path, compression="zstd")
    stats = processed_stats(frame)
    write_json(output_path.with_suffix(".stats.json"), stats)
    return stats


def _write_split_frame_outputs(frame: pd.DataFrame, output_dir: Path, stats: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frame = frame[frame["split"] == split_name]
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        split_frame["major_label"].value_counts().rename_axis("major_label").reset_index(
            name="rows"
        ).to_csv(output_dir / f"{split_name}.class_distribution.csv", index=False)
        write_json(output_dir / f"{split_name}.validate.json", validation_summary(split_frame))
    write_json(output_dir / "split.stats.json", stats)
    return stats


def _write_random_split_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    max_per_major: int,
    seed: int,
) -> dict:
    sampled = stratified_sample(
        frame,
        stratify_column="major_label",
        max_per_class=max_per_major,
        seed=seed,
    )
    sampled = assign_stratified_hash_split(
        sampled,
        group_column="flow_id",
        stratify_column="source_label",
    )
    stats = split_run_stats(
        input_frame=frame,
        output_frame=sampled,
        split_method="random_flow_hash",
        max_per_class=max_per_major,
        seed=seed,
    )
    return _write_split_frame_outputs(sampled, output_dir, stats)


def _write_time_ordered_split_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    max_per_major: int,
    seed: int,
) -> dict:
    sampled = stratified_sample(
        frame,
        stratify_column="major_label",
        max_per_class=max_per_major,
        seed=seed,
    )
    sampled = assign_time_ordered_split(
        sampled,
        group_column="flow_id",
        stratify_column="source_label",
        time_column="start_time",
    )
    stats = split_run_stats(
        input_frame=frame,
        output_frame=sampled,
        split_method="time_ordered",
        max_per_class=max_per_major,
        time_column="start_time",
        seed=seed,
    )
    return _write_split_frame_outputs(sampled, output_dir, stats)


def _write_time_block_split_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    max_per_major: int,
    block_size: int,
    seed: int,
) -> dict:
    sampled = stratified_sample(
        frame,
        stratify_column="major_label",
        max_per_class=max_per_major,
        seed=seed,
    )
    required_group_columns = {
        "bytes",
        "packet_lengths",
        "packet_directions",
        "protocol",
        "endpoint_a",
        "endpoint_b",
        "start_time",
    }
    if required_group_columns <= set(sampled.columns):
        sampled = sampled.copy()
        _ensure_model_visible_hash_columns(sampled)
        sampled["_formal_time_block_group"] = _load_or_build_formal_groups(
            sampled,
            output_dir,
        )
        group_column = "_formal_time_block_group"
    else:
        group_column = "flow_id"
    sampled = assign_time_block_split(
        sampled,
        group_column=group_column,
        stratify_column="source_label",
        time_column="start_time",
        block_size=block_size,
        seed=seed,
    )
    stats = split_run_stats(
        input_frame=frame,
        output_frame=sampled,
        split_method="time_block",
        max_per_class=max_per_major,
        group_column=group_column,
        time_column="start_time",
        block_size=block_size,
        seed=seed,
    )
    return _write_split_frame_outputs(sampled, output_dir, stats)


def _write_submode_stratified_group_split_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    max_per_major: int,
    seed: int,
    low_support_min: int,
) -> dict:
    sampled = stratified_sample(
        frame,
        stratify_column="major_label",
        max_per_class=max_per_major,
        seed=seed,
    )
    required_group_columns = {
        "bytes",
        "packet_lengths",
        "packet_directions",
        "protocol",
        "endpoint_a",
        "endpoint_b",
        "start_time",
        "connection_type",
        "payload_byte_length",
        "source_label",
    }
    if required_group_columns <= set(sampled.columns):
        sampled = sampled.copy()
        _ensure_model_visible_hash_columns(sampled)
        if "_formal_time_block_group" not in sampled.columns:
            sampled["_formal_time_block_group"] = _load_or_build_formal_groups(
                sampled,
                output_dir,
            )
        group_column = "_formal_time_block_group"
        sampled["_submode_signature"] = [
            _submode_signature_id(_submode_signature(row))
            for row in sampled[
                [
                    "packet_lengths",
                    "packet_directions",
                    "connection_type",
                    "payload_byte_length",
                ]
            ].itertuples(index=False)
        ]
        source_counts = sampled["source_label"].value_counts(dropna=False).to_dict()
        min_submode_count = max(low_support_min * 3, 20)
        sampled["_source_submode"] = [
            str(source_label)
            if int(source_counts.get(source_label, 0)) < min_submode_count
            else f"{source_label}::{signature}"
            for source_label, signature in zip(
                sampled["source_label"],
                sampled["_submode_signature"],
                strict=True,
            )
        ]
        stratify_column = "_source_submode"
    else:
        sampled = sampled.copy()
        group_column = "flow_id"
        stratify_column = "source_label"
    sampled = assign_stratified_hash_split(
        sampled,
        group_column=group_column,
        stratify_column=stratify_column,
    )
    if {"source_label", group_column} <= set(sampled.columns):
        sampled = _rebalance_low_support_source_labels(
            sampled,
            group_column=group_column,
            low_support_min=low_support_min,
            seed=seed,
        )
    stats = split_run_stats(
        input_frame=frame,
        output_frame=sampled,
        split_method="submode_stratified_group",
        max_per_class=max_per_major,
        group_column=group_column,
        stratify_column=stratify_column,
        seed=seed,
    )
    stats["low_support_min"] = int(low_support_min)
    return _write_split_frame_outputs(sampled, output_dir, stats)


def _rebalance_low_support_source_labels(
    frame: pd.DataFrame,
    *,
    group_column: str,
    low_support_min: int,
    seed: int,
    max_rebalanced_rows: int = 50,
) -> pd.DataFrame:
    if low_support_min <= 0:
        return frame
    output = frame.copy()
    source_counts = output["source_label"].value_counts(dropna=False).to_dict()
    for source_label, total_rows in source_counts.items():
        source_text = str(source_label)
        if source_text.upper() == "BENIGN":
            continue
        total_rows = int(total_rows)
        if total_rows < low_support_min * 3 or total_rows > max_rebalanced_rows:
            continue
        label_mask = output["source_label"].astype(str).eq(source_text)
        group_sizes = (
            output.loc[label_mask, [group_column]]
            .groupby(group_column, dropna=False)
            .size()
            .reset_index(name="rows")
        )
        if len(group_sizes) < 3:
            continue
        group_sizes["_bucket"] = [
            stable_bucket(f"{seed}:{source_text}:{group}")
            for group in group_sizes[group_column]
        ]
        group_sizes = group_sizes.sort_values(["_bucket", group_column], kind="mergesort")
        assigned_rows = {"train": 0, "val": 0, "test": 0}
        group_to_split: dict[str, str] = {}
        eval_targets = {"val": low_support_min, "test": low_support_min}
        for group_value, rows, _bucket in group_sizes[
            [group_column, "rows", "_bucket"]
        ].itertuples(index=False, name=None):
            group = str(group_value)
            rows = int(rows)
            if assigned_rows["val"] < eval_targets["val"]:
                split = "val"
            elif assigned_rows["test"] < eval_targets["test"]:
                split = "test"
            else:
                split = "train"
            group_to_split[group] = split
            assigned_rows[split] += rows
        output.loc[label_mask, "split"] = [
            group_to_split[str(group)] for group in output.loc[label_mask, group_column]
        ]
    return output


def _canonical_tuple(protocol: object, endpoint_a: object, endpoint_b: object) -> tuple[str, str, str]:
    left, right = sorted([str(endpoint_a), str(endpoint_b)])
    return str(protocol), left, right


def _ensure_model_visible_hash_columns(frame: pd.DataFrame) -> None:
    required = {"bytes", "packet_lengths", "packet_directions"}
    if not required <= set(frame.columns):
        return
    if "_bytes_sha1" not in frame.columns:
        frame["_bytes_sha1"] = frame["bytes"].map(_bytes_sha1)
    if "_first_window_sha1" not in frame.columns:
        frame["_first_window_sha1"] = [
            _first_window_sha1(row.bytes, row.packet_lengths, row.packet_directions)
            for row in frame[["bytes", "packet_lengths", "packet_directions"]].itertuples(
                index=False
            )
        ]


def _formal_group_cache_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("flow_id", "view", "source_label", "start_time")
        if column in frame.columns
    ]
    if not columns:
        columns = list(frame.columns[:1])
    digest = hashlib.sha1()
    digest.update(FORMAL_GROUP_CACHE_VERSION.encode("utf-8"))
    digest.update(str(len(frame)).encode("utf-8"))
    for row in frame[columns].itertuples(index=False, name=None):
        for value in row:
            digest.update(str(value).encode("utf-8", errors="replace"))
            digest.update(b"\x1f")
        digest.update(b"\x1e")
    return digest.hexdigest()


def _formal_group_cache_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "_formal_time_block_groups.cache.parquet",
        output_dir / "_formal_time_block_groups.cache.json",
    )


def _load_formal_group_cache(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    nearby_threshold_seconds: float,
) -> list[str] | None:
    cache_path, meta_path = _formal_group_cache_paths(output_dir)
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("version") != FORMAL_GROUP_CACHE_VERSION:
        return None
    if int(metadata.get("rows", -1)) != len(frame):
        return None
    if float(metadata.get("nearby_threshold_seconds", -1)) != float(nearby_threshold_seconds):
        return None
    started = time.monotonic()
    fingerprint = _formal_group_cache_fingerprint(frame)
    if metadata.get("fingerprint") != fingerprint:
        return None
    try:
        cached = pd.read_parquet(cache_path, columns=["group"])
    except (OSError, ValueError, KeyError):
        return None
    if len(cached) != len(frame):
        return None
    _log_event(
        "formal_group_cache_hit",
        cache_path=str(cache_path),
        rows=len(frame),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return [str(value) for value in cached["group"]]


def _write_formal_group_cache(
    frame: pd.DataFrame,
    output_dir: Path,
    groups: list[str],
    *,
    nearby_threshold_seconds: float,
) -> None:
    cache_path, meta_path = _formal_group_cache_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pd.DataFrame({"group": groups}).to_parquet(cache_path, index=False)
    metadata = {
        "version": FORMAL_GROUP_CACHE_VERSION,
        "rows": len(frame),
        "nearby_threshold_seconds": float(nearby_threshold_seconds),
        "fingerprint": _formal_group_cache_fingerprint(frame),
    }
    write_json(meta_path, metadata)
    _log_event(
        "formal_group_cache_write_done",
        cache_path=str(cache_path),
        rows=len(frame),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


def _load_or_build_formal_groups(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    nearby_threshold_seconds: float = 120.0,
) -> list[str]:
    cached = _load_formal_group_cache(
        frame,
        output_dir,
        nearby_threshold_seconds=nearby_threshold_seconds,
    )
    if cached is not None:
        return cached
    groups = _formal_time_block_groups(
        frame,
        nearby_threshold_seconds=nearby_threshold_seconds,
    )
    _write_formal_group_cache(
        frame,
        output_dir,
        groups,
        nearby_threshold_seconds=nearby_threshold_seconds,
    )
    return groups


def _formal_time_block_groups(
    frame: pd.DataFrame,
    *,
    nearby_threshold_seconds: float = 120.0,
) -> list[str]:
    started = time.monotonic()
    _log_event(
        "formal_group_build_start",
        rows=len(frame),
        nearby_threshold_seconds=nearby_threshold_seconds,
    )
    parent = list(range(len(frame)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[str, int] = {}
    duplicate_hash_edges = 0
    if {"_bytes_sha1", "_first_window_sha1"} <= set(frame.columns):
        hash_iter = (
            (idx, f"bytes:{bytes_hash}", f"first:{first_hash}")
            for idx, (bytes_hash, first_hash) in enumerate(
                frame[["_bytes_sha1", "_first_window_sha1"]].itertuples(
                    index=False,
                    name=None,
                )
            )
        )
    else:
        hash_iter = (
            (
                idx,
                f"bytes:{_bytes_sha1(row.bytes)}",
                "first:"
                + _first_window_sha1(row.bytes, row.packet_lengths, row.packet_directions),
            )
            for idx, row in enumerate(
                frame[["bytes", "packet_lengths", "packet_directions"]].itertuples(
                    index=False
                )
            )
        )
    for idx, bytes_digest, first_window_digest in hash_iter:
        hashes = (bytes_digest, first_window_digest)
        for digest in hashes:
            previous = seen.get(digest)
            if previous is None:
                seen[digest] = idx
            else:
                union(previous, idx)
                duplicate_hash_edges += 1
    _log_event(
        "formal_group_hash_pass_done",
        rows=len(frame),
        unique_hashes=len(seen),
        duplicate_edges=duplicate_hash_edges,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )

    tuple_started = time.monotonic()
    tuple_work = frame[["protocol", "endpoint_a", "endpoint_b", "start_time"]].copy()
    tuple_work["_idx"] = range(len(frame))
    tuple_work = tuple_work.dropna(subset=["start_time"])
    endpoint_a = tuple_work["endpoint_a"].astype(str)
    endpoint_b = tuple_work["endpoint_b"].astype(str)
    left = endpoint_a.where(endpoint_a <= endpoint_b, endpoint_b)
    right = endpoint_b.where(endpoint_a <= endpoint_b, endpoint_a)
    tuple_work["_tuple"] = tuple_work["protocol"].astype(str) + "|" + left + "|" + right
    tuple_work = tuple_work.sort_values(["_tuple", "start_time", "_idx"], kind="mergesort")

    times = pd.to_numeric(tuple_work["start_time"], errors="coerce")
    same_tuple = tuple_work["_tuple"].eq(tuple_work["_tuple"].shift())
    nearby = same_tuple & times.diff().le(float(nearby_threshold_seconds))
    current_indexes = tuple_work.loc[nearby, "_idx"].astype(int).to_numpy()
    previous_indexes = tuple_work["_idx"].shift().loc[nearby].astype(int).to_numpy()
    for previous_idx, current_idx in zip(previous_indexes, current_indexes, strict=True):
        union(int(previous_idx), int(current_idx))
    _log_event(
        "formal_group_tuple_pass_done",
        rows=len(tuple_work),
        nearby_edges=len(current_indexes),
        elapsed_seconds=round(time.monotonic() - tuple_started, 3),
    )

    groups = [f"formal-{find(idx)}" for idx in range(len(frame))]
    _log_event(
        "formal_group_build_done",
        rows=len(frame),
        groups=len(set(groups)),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return groups


def _has_low_support(stats: dict, low_support_min: int) -> bool:
    support = stats.get("split_support", {}).get("major_labels", {})
    for label in ["infiltration", "web_attack", "botnet_malware"]:
        for split_name in ["val", "test"]:
            if int(support.get(split_name, {}).get(label, 0)) < low_support_min:
                return True
    return False


def _validate_attack_coverage(
    summaries: list[dict],
    shard_paths: list[Path],
    min_attack_flows_per_label: int,
    min_attack_match_ratio: float,
) -> dict:
    failures = []
    report = {
        "formal_eligible": True,
        "min_attack_flows_per_label": int(min_attack_flows_per_label),
        "min_attack_match_ratio": float(min_attack_match_ratio),
        "items": [],
        "warnings": [],
        "failures": failures,
    }
    for summary, shard_path in zip(summaries, shard_paths, strict=True):
        attack_labels = summary.get("attack_labels", [])
        if not attack_labels:
            continue
        if not shard_path.exists():
            failures.append(
                f"{summary['slug']}: missing processed shard {shard_path}"
            )
            continue
        schema_names = pq.read_schema(shard_path).names
        frame = None
        major_counts = {}
        if "source_label" not in schema_names:
            counts = {}
        else:
            columns = ["source_label"]
            for column in ["major_label", "label_match_mode", "connection_type"]:
                if column in schema_names:
                    columns.append(column)
            frame = pd.read_parquet(shard_path, columns=columns)
            counts = frame["source_label"].value_counts().to_dict()
            if "major_label" in frame:
                major_counts = frame["major_label"].value_counts().to_dict()
        for label in attack_labels:
            official_count = int(summary.get("label_counts", {}).get(label, 0))
            count = int(counts.get(label, 0))
            if official_count <= 0:
                failures.append(f"{summary['slug']} / {label}: no official label rows")
                continue
            ratio = count / official_count
            low_support_exception = official_count < min_attack_flows_per_label
            required_count = (
                max(1, math.ceil(official_count * min_attack_match_ratio))
                if low_support_exception
                else min_attack_flows_per_label
            )
            label_major = None
            processed_major_flows = 0
            label_match_mode_counts = {}
            label_connection_type_counts = {}
            if frame is not None and "major_label" in frame and count > 0:
                label_rows = frame[frame["source_label"] == label]
                if not label_rows.empty:
                    label_major = str(label_rows["major_label"].mode().iloc[0])
                    processed_major_flows = int(major_counts.get(label_major, 0))
                    if "label_match_mode" in label_rows:
                        label_match_mode_counts = label_rows["label_match_mode"].value_counts(
                            dropna=False
                        ).to_dict()
                    if "connection_type" in label_rows:
                        label_connection_type_counts = label_rows["connection_type"].value_counts(
                            dropna=False
                        ).to_dict()
            major_support_exception = (
                not low_support_exception
                and count < required_count
                and ratio >= min_attack_match_ratio
                and processed_major_flows >= min_attack_flows_per_label
            )
            item = {
                "slug": summary["slug"],
                "label": label,
                "official_label_rows": official_count,
                "processed_flows": count,
                "match_ratio": ratio,
                "required_flows": required_count,
                "low_support_exception": low_support_exception,
                "major_label": label_major,
                "processed_major_flows": processed_major_flows,
                "major_support_exception": major_support_exception,
                "label_match_modes": {
                    str(key): int(value) for key, value in label_match_mode_counts.items()
                },
                "connection_types": {
                    str(key): int(value) for key, value in label_connection_type_counts.items()
                },
            }
            report["items"].append(item)
            if low_support_exception:
                report["warnings"].append(
                    f"{summary['slug']} / {label}: official support {official_count} "
                    f"< {min_attack_flows_per_label}; using ratio-derived minimum {required_count}"
                )
            if major_support_exception:
                report["warnings"].append(
                    f"{summary['slug']} / {label}: source-label support {count} "
                    f"< {required_count}, but major label {label_major!r} has "
                    f"{processed_major_flows} processed flows and match ratio {ratio:.6f} "
                    f">= {min_attack_match_ratio:.6f}; treat this source/minor label "
                    "as low-support diagnostic only"
                )
            elif count < required_count:
                failures.append(
                    f"{summary['slug']} / {label}: {count} processed flows "
                    f"< {required_count}"
                )
            if ratio < min_attack_match_ratio:
                failures.append(
                    f"{summary['slug']} / {label}: match ratio {ratio:.6f} "
                    f"< {min_attack_match_ratio:.6f}"
                )
    if failures:
        report["formal_eligible"] = False
        joined = "\n  - ".join(failures)
        raise AttackCoverageError(
            "CICIDS2017 attack coverage check failed. Refusing to write training "
            "splits from broken coverage:\n  - "
            f"{joined}\n"
            "Check PCAP/CSV timestamp alignment, or rerun with an explicit "
            "--csv-time-offset-hours value.",
            report,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/CICIDS2017"))
    parser.add_argument(
        "--label-zip",
        type=Path,
        default=Path("data/raw/CICIDS2017/csvs/GeneratedLabelledFlows.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose"),
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_label_stratified_cap32_notcpclose"),
    )
    parser.add_argument(
        "--time-ordered-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_time_ordered_cap32_notcpclose"),
    )
    parser.add_argument(
        "--time-block-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_time_block_cap32_notcpclose"),
    )
    parser.add_argument(
        "--submode-stratified-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose"),
    )
    parser.add_argument(
        "--time-block-small-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_time_block_128_cap32_notcpclose"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/cicids2017_coverage_masked_header_cap32_notcpclose"),
    )
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument(
        "--views",
        default="masked_header_packet",
        help="Comma-separated processed views to write. Formal CICIDS uses masked_header_packet.",
    )
    parser.add_argument(
        "--drop-empty-payload",
        action="store_true",
        help="Drop flows without L4 payload. Leave unset for header-aware formal CICIDS builds.",
    )
    parser.add_argument(
        "--allow-nonformal-output",
        action="store_true",
        help="Allow non-formal CICIDS variants, such as payload-only or dropping empty payload flows.",
    )
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-packets-per-flow", type=int, default=32)
    parser.add_argument("--flow-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--close-on-tcp-flags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split TCP sessions when FIN/RST is observed. Formal CICIDS builds keep this "
            "disabled so TCP close/reset packets remain attached to the connection attempt."
        ),
    )
    parser.add_argument(
        "--cic-label-max-time-delta-seconds",
        type=float,
        default=DEFAULT_CIC_LABEL_MAX_TIME_DELTA_SECONDS,
        help=(
            "Maximum absolute time difference allowed when matching a reconstructed "
            "PCAP flow to duplicate CIC labelled-flow rows with the same five-tuple."
        ),
    )
    parser.add_argument("--max-packets-to-read", type=int, default=None)
    parser.add_argument("--padding-minutes", type=int, default=20)
    parser.add_argument(
        "--csv-time-offset-hours",
        type=float,
        default=3.0,
        help="Hours added to official CSV timestamps to align local CICIDS time with PCAP UTC.",
    )
    parser.add_argument(
        "--max-per-major",
        type=int,
        default=0,
        help=(
            "Maximum rows per major class when writing split parquet files. "
            "Use 0 for no cap; formal full-data builds leave this unset."
        ),
    )
    parser.add_argument("--min-attack-flows-per-label", type=int, default=100)
    parser.add_argument("--min-attack-match-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-block-size", type=int, default=512)
    parser.add_argument("--low-support-min", type=int, default=2)
    parser.add_argument(
        "--formal-only-splits",
        action="store_true",
        help="Only write the formal submode-stratified split; skip random/time stress splits.",
    )
    parser.add_argument(
        "--window-scope",
        choices=("pcap", "label", "attack"),
        default="pcap",
        help=(
            "Packet range to scan for each shard. Formal CICIDS builds default to "
            "pcap so time is not used to crop training data."
        ),
    )
    parser.add_argument("--no-time-window", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_views = {item.strip() for item in args.views.split(",") if item.strip()}
    if not args.allow_nonformal_output:
        if "masked_header_packet" not in requested_views:
            raise ValueError(
                "Formal CICIDS preprocessing requires masked_header_packet. "
                "Pass --allow-nonformal-output only for explicit diagnostics."
            )
        if args.drop_empty_payload:
            raise ValueError(
                "Formal CICIDS preprocessing must keep empty-payload TCP control flows. "
                "Pass --allow-nonformal-output only for explicit diagnostics."
            )
        if args.window_scope != "pcap" or args.no_time_window:
            raise ValueError(
                "Formal CICIDS preprocessing must scan full PCAPs with --window-scope pcap. "
                "Pass --allow-nonformal-output only for explicit diagnostics."
            )
        if args.close_on_tcp_flags:
            raise ValueError(
                "Formal CICIDS preprocessing must keep --no-close-on-tcp-flags. "
                "Immediate FIN/RST splitting creates low-information TCP-control fragments, "
                "especially for Bot. Pass --allow-nonformal-output only for diagnostics."
            )
        if float(args.csv_time_offset_hours) != 3.0:
            raise ValueError(
                "Formal CICIDS preprocessing in this repo expects --csv-time-offset-hours 3.0. "
                "Run alignment diagnostics before changing this value, and use "
                "--allow-nonformal-output only for explicit diagnostics."
            )
    if not args.label_zip.exists():
        raise FileNotFoundError(f"missing label zip: {args.label_zip}")
    _validate_required_pcaps(args.raw_dir)

    summaries = [
        _extract_label_file(
            raw_dir=args.raw_dir,
            label_zip=args.label_zip,
            job=job,
            padding_minutes=args.padding_minutes,
            csv_time_offset_hours=args.csv_time_offset_hours,
            window_scope="pcap" if args.no_time_window else args.window_scope,
        )
        for job in JOBS
    ]
    coverage = build_cicids_coverage_audit(
        raw_dir=args.raw_dir,
        label_zip=args.label_zip,
        label_map_path=args.label_map,
        processed_dir=args.output_dir,
    )
    write_cicids_coverage_artifacts(coverage, args.audit_dir)

    payloads = []
    for job, summary in zip(JOBS, summaries, strict=True):
        pcap_path = _resolve_pcap(args.raw_dir, job.pcap_name)
        output_path = args.output_dir / f"flows_{job.slug}.parquet"
        payload = {
            "pcap_path": str(pcap_path),
            "output_path": str(output_path),
            "label_map": str(args.label_map),
            "label_csv": summary["output_path"],
            "max_packets_per_flow": args.max_packets_per_flow,
            "max_packets_to_read": args.max_packets_to_read,
            "flow_timeout_seconds": args.flow_timeout_seconds,
            "close_on_tcp_flags": args.close_on_tcp_flags,
            "cic_label_max_time_delta_seconds": args.cic_label_max_time_delta_seconds,
            "views": [item.strip() for item in args.views.split(",") if item.strip()],
            "keep_empty_payload": not args.drop_empty_payload,
            "min_packet_time": None if args.no_time_window else summary["window_start_unix"],
            "max_packet_time": None if args.no_time_window else summary["window_end_unix"],
            "csv_time_offset_hours": args.csv_time_offset_hours,
            "cicids_timestamp_policy_version": TIMESTAMP_POLICY_VERSION,
            "window_scope": "pcap" if args.no_time_window else args.window_scope,
        }
        if not args.force and _is_fresh_shard(output_path, payload):
            print(f"skip existing fresh shard: {output_path}")
            continue
        if output_path.exists():
            print(f"rebuild stale shard: {output_path}")
        payloads.append(payload)

    built = []
    if payloads:
        grouped_payloads: dict[tuple, list[dict]] = {}
        for payload in payloads:
            grouped_payloads.setdefault(_extract_key(payload), []).append(payload)
        if len(grouped_payloads) < len(payloads):
            print(
                "Grouped CICIDS build payloads by PCAP extraction parameters: "
                f"{len(payloads)} shard builds -> {len(grouped_payloads)} PCAP scans"
            )
        with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [
                executor.submit(_build_group, group)
                for group in grouped_payloads.values()
            ]
            for future in as_completed(futures):
                for result in future.result():
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    built.append(result)

    coverage = build_cicids_coverage_audit(
        raw_dir=args.raw_dir,
        label_zip=args.label_zip,
        label_map_path=args.label_map,
        processed_dir=args.output_dir,
    )
    write_cicids_coverage_artifacts(coverage, args.audit_dir)

    shard_paths = [args.output_dir / f"flows_{job.slug}.parquet" for job in JOBS]
    try:
        attack_coverage = _validate_attack_coverage(
            summaries,
            shard_paths,
            min_attack_flows_per_label=args.min_attack_flows_per_label,
            min_attack_match_ratio=args.min_attack_match_ratio,
        )
    except AttackCoverageError as exc:
        write_json(args.audit_dir / "attack_coverage.json", exc.report)
        raise
    write_json(args.audit_dir / "attack_coverage.json", attack_coverage)
    merged_stats = _merge_outputs(shard_paths, args.merged_path)
    merged_frame = pd.read_parquet(args.merged_path)
    split_stats = None
    time_ordered_stats = None
    time_block_stats = None
    if not args.formal_only_splits:
        split_stats = _write_random_split_outputs(
            merged_frame,
            args.split_dir,
            args.max_per_major,
            args.seed,
        )
        time_ordered_stats = _write_time_ordered_split_outputs(
            merged_frame,
            args.time_ordered_split_dir,
            args.max_per_major,
            args.seed,
        )
        time_block_stats = _write_time_block_split_outputs(
            merged_frame,
            args.time_block_split_dir,
            args.max_per_major,
            args.time_block_size,
            args.seed,
        )
    submode_stratified_stats = _write_submode_stratified_group_split_outputs(
        merged_frame,
        args.submode_stratified_split_dir,
        args.max_per_major,
        args.seed,
        args.low_support_min,
    )
    time_block_small_stats = None
    if (
        time_block_stats is not None
        and _has_low_support(time_block_stats, args.low_support_min)
    ):
        time_block_small_stats = _write_time_block_split_outputs(
            merged_frame,
            args.time_block_small_split_dir,
            args.max_per_major,
            128,
            args.seed,
        )
    print(
        json.dumps(
            {
                "labels": summaries,
                "coverage_audit_dir": str(args.audit_dir),
                "attack_coverage": attack_coverage,
                "built": built,
                "merged_path": str(args.merged_path),
                "merged_stats": merged_stats,
                "split_dir": str(args.split_dir),
                "split_stats": split_stats,
                "time_ordered_split_dir": str(args.time_ordered_split_dir),
                "time_ordered_stats": time_ordered_stats,
                "time_block_split_dir": str(args.time_block_split_dir),
                "time_block_stats": time_block_stats,
                "submode_stratified_split_dir": str(args.submode_stratified_split_dir),
                "submode_stratified_stats": submode_stratified_stats,
                "time_block_small_split_dir": str(args.time_block_small_split_dir)
                if time_block_small_stats is not None
                else None,
                "time_block_small_stats": time_block_small_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
