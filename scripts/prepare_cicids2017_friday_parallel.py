"""Prepare CICIDS2017 Friday attack windows in parallel.

This script builds separate processed Parquet files for Friday Morning Bot,
Friday Afternoon PortScan, and Friday Afternoon DDoS, then merges them into a
single split suitable for sanity training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timezone
import json
from pathlib import Path
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_bert.config import write_json
from traffic_bert.data.build import BuildConfig, build_processed_dataset
from traffic_bert.data.schema import InputView
from traffic_bert.data.split import (
    assign_stratified_hash_split,
    processed_stats,
    split_run_stats,
    stratified_sample,
)
from traffic_bert.data.validate import validation_summary


@dataclass(frozen=True)
class FridayJob:
    slug: str
    filename_contains: str
    attack_labels: tuple[str, ...]


JOBS = (
    FridayJob(
        slug="bot",
        filename_contains="Morning",
        attack_labels=("Bot",),
    ),
    FridayJob(
        slug="portscan",
        filename_contains="PortScan",
        attack_labels=("PortScan",),
    ),
    FridayJob(
        slug="ddos",
        filename_contains="DDos",
        attack_labels=("DDoS",),
    ),
)


def _find_csv_name(label_zip: Path, filename_contains: str) -> str:
    needle = filename_contains.lower()
    with zipfile.ZipFile(label_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and "friday" in Path(name).name.lower()
            and needle in Path(name).name.lower()
        ]
    if not matches:
        raise FileNotFoundError(
            f"no Friday CSV containing {filename_contains!r} in {label_zip}"
        )
    if len(matches) > 1:
        raise ValueError(f"ambiguous Friday CSVs for {filename_contains!r}: {matches}")
    return matches[0]


def _read_label_csv(label_zip: Path, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(label_zip) as archive:
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, encoding="latin1", low_memory=False)


def _label_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if column.strip().lower() == "label":
            return column
    raise ValueError("label CSV has no Label column")


def _timestamp_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if column.strip().lower() == "timestamp":
            return column
    raise ValueError("label CSV has no Timestamp column")


def _unix_seconds(value: pd.Timestamp) -> float:
    if value.tzinfo is None:
        value = value.tz_localize(timezone.utc)
    return float(value.timestamp())


def _parse_cicids_timestamps(values: pd.Series, csv_name: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if "afternoon" not in Path(csv_name).name.lower():
        return timestamps

    valid = timestamps.notna()
    morning_like = valid & (timestamps.dt.hour < 12)
    timestamps.loc[morning_like] = timestamps.loc[morning_like] + pd.Timedelta(hours=12)
    return timestamps


def _extract_label_file(
    *,
    label_zip: Path,
    labels_dir: Path,
    job: FridayJob,
    padding_minutes: int,
    csv_time_offset_hours: float,
) -> dict:
    csv_name = _find_csv_name(label_zip, job.filename_contains)
    frame = _read_label_csv(label_zip, csv_name)
    label_col = _label_column(frame)
    ts_col = _timestamp_column(frame)

    timestamps = _parse_cicids_timestamps(frame[ts_col], csv_name)
    original_min_time = timestamps.dropna().min()
    original_max_time = timestamps.dropna().max()
    if csv_time_offset_hours:
        timestamps = timestamps + pd.Timedelta(hours=csv_time_offset_hours)
    frame[ts_col] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_path = labels_dir / f"friday_labelled_flows_{job.slug}.csv"
    frame.to_csv(output_path, index=False)
    attack_mask = frame[label_col].isin(job.attack_labels)
    attack_times = timestamps[attack_mask & timestamps.notna()]
    if attack_times.empty:
        min_time = timestamps.dropna().min()
        max_time = timestamps.dropna().max()
    else:
        min_time = attack_times.min() - pd.Timedelta(minutes=padding_minutes)
        max_time = attack_times.max() + pd.Timedelta(minutes=padding_minutes)

    summary = {
        "slug": job.slug,
        "csv_name": csv_name,
        "output_path": str(output_path),
        "rows": int(len(frame)),
        "label_counts": frame[label_col].value_counts().to_dict(),
        "attack_labels": list(job.attack_labels),
        "csv_time_offset_hours": csv_time_offset_hours,
        "original_timestamp_start": None if pd.isna(original_min_time) else str(original_min_time),
        "original_timestamp_end": None if pd.isna(original_max_time) else str(original_max_time),
        "window_start": None if pd.isna(min_time) else str(min_time),
        "window_end": None if pd.isna(max_time) else str(max_time),
        "window_start_unix": None if pd.isna(min_time) else _unix_seconds(min_time),
        "window_end_unix": None if pd.isna(max_time) else _unix_seconds(max_time),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _build_one(payload: dict) -> dict:
    config = BuildConfig(
        input_path=Path(payload["pcap_path"]),
        output_path=Path(payload["output_path"]),
        label_map_path=Path(payload["label_map"]),
        source_dataset="cicids2017-friday",
        split="train",
        label_source="cic_csv",
        label_csv_path=Path(payload["label_csv"]),
        drop_unmatched_labels=True,
        views=(InputView.PAYLOAD_ONLY,),
        keep_empty_payload=False,
        max_packets_per_flow=int(payload["max_packets_per_flow"]),
        max_packets_to_read=payload["max_packets_to_read"],
        min_packet_time=payload["min_packet_time"],
        max_packet_time=payload["max_packet_time"],
    )
    stats = build_processed_dataset(config)
    build_payload = {
        **stats,
        "label_csv": str(config.label_csv_path),
        "min_packet_time": config.min_packet_time,
        "max_packet_time": config.max_packet_time,
        "max_packets_to_read": config.max_packets_to_read,
        "max_packets_per_flow": config.max_packets_per_flow,
    }
    write_json(config.output_path.with_suffix(".build.json"), build_payload)
    return {"output_path": str(config.output_path), "stats": stats}


def _merge_outputs(paths: list[Path], output_path: Path) -> dict:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("no processed Friday shard outputs were created")
    frame = pd.concat(frames, ignore_index=True)
    dedupe_columns = [
        column
        for column in ["flow_id", "view", "source_label", "major_label"]
        if column in frame.columns
    ]
    frame = frame.drop_duplicates(subset=dedupe_columns).sort_values(
        ["start_time", "flow_id"], kind="mergesort"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")
    stats = processed_stats(frame)
    write_json(output_path.with_suffix(".stats.json"), stats)
    return stats


def _write_split_outputs(
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
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frame = sampled[sampled["split"] == split_name]
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        split_frame["major_label"].value_counts().rename_axis("major_label").reset_index(
            name="rows"
        ).to_csv(output_dir / f"{split_name}.class_distribution.csv", index=False)
        write_json(
            output_dir / f"{split_name}.validate.json",
            validation_summary(split_frame),
        )
    stats = split_run_stats(
        input_frame=frame,
        output_frame=sampled,
        split_method="random_flow_hash",
        max_per_class=max_per_major,
        seed=seed,
    )
    write_json(output_dir / "split.stats.json", stats)
    return stats


def _validate_attack_coverage(
    summaries: list[dict],
    shard_paths: list[Path],
    min_attack_flows_per_label: int,
) -> None:
    failures = []
    for summary, shard_path in zip(summaries, shard_paths, strict=True):
        attack_labels = summary.get("attack_labels", [])
        if not attack_labels:
            continue
        if not shard_path.exists():
            failures.append(f"{summary['slug']}: missing processed shard {shard_path}")
            continue
        if "source_label" not in pq.read_schema(shard_path).names:
            counts = {}
        else:
            frame = pd.read_parquet(shard_path, columns=["source_label"])
            counts = frame["source_label"].value_counts().to_dict()
        for label in attack_labels:
            count = int(counts.get(label, 0))
            if count < min_attack_flows_per_label:
                failures.append(
                    f"{summary['slug']} / {label}: {count} processed flows "
                    f"< {min_attack_flows_per_label}"
                )
    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(
            "CICIDS2017 Friday attack coverage check failed:\n  - "
            f"{joined}\n"
            "Check PCAP/CSV timestamp alignment, or rerun with an explicit "
            "--csv-time-offset-hours value."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/CICIDS2017"),
    )
    parser.add_argument(
        "--pcap-path",
        type=Path,
        default=Path("data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap"),
    )
    parser.add_argument(
        "--label-zip",
        type=Path,
        default=Path("data/raw/CICIDS2017/csvs/GeneratedLabelledFlows.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cicids2017/friday_payload_only"),
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=Path("data/processed/cicids2017/friday_payload_only/flows_friday_attacks.parquet"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/friday_split_label_stratified"),
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("configs/label_map.yaml"),
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-packets-per-flow", type=int, default=16)
    parser.add_argument(
        "--max-packets-to-read",
        type=int,
        default=None,
        help="Optional cap per attack window after the window start.",
    )
    parser.add_argument("--padding-minutes", type=int, default=20)
    parser.add_argument(
        "--csv-time-offset-hours",
        type=float,
        default=4.0,
        help="Hours added to official CSV timestamps to align local CICIDS time with PCAP UTC.",
    )
    parser.add_argument("--max-per-major", type=int, default=50_000)
    parser.add_argument("--min-attack-flows-per-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-time-window",
        action="store_true",
        help="Scan the full PCAP for each Friday CSV instead of attack-time windows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild shard parquet files even when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pcap_path.exists():
        raise FileNotFoundError(f"missing PCAP: {args.pcap_path}")
    if not args.label_zip.exists():
        raise FileNotFoundError(f"missing label zip: {args.label_zip}")

    labels_dir = args.raw_dir / "labels"
    label_summaries = [
        _extract_label_file(
            label_zip=args.label_zip,
            labels_dir=labels_dir,
            job=job,
            padding_minutes=args.padding_minutes,
            csv_time_offset_hours=args.csv_time_offset_hours,
        )
        for job in JOBS
    ]

    build_payloads = []
    for summary in label_summaries:
        slug = summary["slug"]
        output_path = args.output_dir / f"flows_{slug}_formal.parquet"
        if output_path.exists() and not args.force:
            print(f"skip existing: {output_path}")
            continue
        build_payloads.append(
            {
                "pcap_path": str(args.pcap_path),
                "output_path": str(output_path),
                "label_map": str(args.label_map),
                "label_csv": summary["output_path"],
                "max_packets_per_flow": args.max_packets_per_flow,
                "max_packets_to_read": args.max_packets_to_read,
                "min_packet_time": None
                if args.no_time_window
                else summary["window_start_unix"],
                "max_packet_time": None
                if args.no_time_window
                else summary["window_end_unix"],
            }
        )

    results = []
    if build_payloads:
        with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [executor.submit(_build_one, payload) for payload in build_payloads]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result, ensure_ascii=False, indent=2))
                results.append(result)

    shard_paths = [args.output_dir / f"flows_{job.slug}_formal.parquet" for job in JOBS]
    _validate_attack_coverage(
        label_summaries,
        shard_paths,
        min_attack_flows_per_label=args.min_attack_flows_per_label,
    )
    merged_stats = _merge_outputs(shard_paths, args.merged_path)
    split_stats = _write_split_outputs(
        pd.read_parquet(args.merged_path),
        args.split_dir,
        max_per_major=args.max_per_major,
        seed=args.seed,
    )
    final = {
        "labels": label_summaries,
        "built": results,
        "merged_path": str(args.merged_path),
        "merged_stats": merged_stats,
        "split_dir": str(args.split_dir),
        "split_stats": split_stats,
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
