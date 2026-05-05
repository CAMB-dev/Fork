"""Prepare CICIDS2017 working-hour PCAPs in parallel.

The script builds one processed Parquet shard per official labelled-flow CSV,
then merges all shards and writes a stratified train/val/test split.
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
from traffic_bert.data.cicids_audit import (
    build_cicids_coverage_audit,
    write_cicids_coverage_artifacts,
)
from traffic_bert.data.build import BuildConfig, build_processed_dataset
from traffic_bert.data.schema import InputView
from traffic_bert.data.split import (
    assign_time_block_split,
    assign_time_ordered_split,
    assign_stratified_hash_split,
    processed_stats,
    split_run_stats,
    stratified_sample,
)
from traffic_bert.data.validate import validation_summary


@dataclass(frozen=True)
class CicidsJob:
    slug: str
    pcap_name: str
    csv_contains: tuple[str, ...]
    attack_labels: tuple[str, ...]


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
    if "afternoon" not in Path(csv_name).name.lower():
        return timestamps

    valid = timestamps.notna()
    morning_like = valid & (timestamps.dt.hour < 12)
    timestamps.loc[morning_like] = timestamps.loc[morning_like] + pd.Timedelta(hours=12)
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
) -> dict:
    csv_name = _find_csv_name(label_zip, job.csv_contains)
    frame = _read_label_csv(label_zip, csv_name)
    label_col = _column(frame, "label")
    ts_col = _column(frame, "timestamp")

    timestamps = _parse_cicids_timestamps(frame[ts_col], csv_name)
    frame[ts_col] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    labels_dir = raw_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_path = labels_dir / f"cicids2017_{job.slug}.csv"
    frame.to_csv(output_path, index=False)
    if job.attack_labels:
        attack_mask = frame[label_col].isin(job.attack_labels)
        selected_times = timestamps[attack_mask & timestamps.notna()]
    else:
        selected_times = timestamps[timestamps.notna()]

    if selected_times.empty:
        min_time = timestamps.dropna().min()
        max_time = timestamps.dropna().max()
    else:
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
        source_dataset="cicids2017",
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
    write_json(
        config.output_path.with_suffix(".build.json"),
        {
            **stats,
            "label_csv": str(config.label_csv_path),
            "min_packet_time": config.min_packet_time,
            "max_packet_time": config.max_packet_time,
            "max_packets_to_read": config.max_packets_to_read,
            "max_packets_per_flow": config.max_packets_per_flow,
        },
    )
    return {"output_path": str(config.output_path), "stats": stats}


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
    sampled = assign_time_block_split(
        sampled,
        group_column="flow_id",
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
        time_column="start_time",
        block_size=block_size,
        seed=seed,
    )
    return _write_split_frame_outputs(sampled, output_dir, stats)


def _has_low_support(stats: dict, low_support_min: int) -> bool:
    support = stats.get("split_support", {}).get("major_labels", {})
    for label in ["infiltration", "web_attack", "botnet_malware"]:
        for split_name in ["val", "test"]:
            if int(support.get(split_name, {}).get(label, 0)) < low_support_min:
                return True
    return False


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
        default=Path("data/processed/cicids2017/all_payload_only"),
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=Path("data/processed/cicids2017/all_payload_only/flows_all.parquet"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_split_label_stratified"),
    )
    parser.add_argument(
        "--time-ordered-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_split_time_ordered"),
    )
    parser.add_argument(
        "--time-block-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_split_time_block"),
    )
    parser.add_argument(
        "--time-block-small-split-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_split_time_block_128"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/cicids2017_coverage"),
    )
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-packets-per-flow", type=int, default=16)
    parser.add_argument("--max-packets-to-read", type=int, default=None)
    parser.add_argument("--padding-minutes", type=int, default=20)
    parser.add_argument("--max-per-major", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-block-size", type=int, default=512)
    parser.add_argument("--low-support-min", type=int, default=2)
    parser.add_argument("--no-time-window", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.label_zip.exists():
        raise FileNotFoundError(f"missing label zip: {args.label_zip}")

    summaries = [
        _extract_label_file(
            raw_dir=args.raw_dir,
            label_zip=args.label_zip,
            job=job,
            padding_minutes=args.padding_minutes,
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
        if output_path.exists() and not args.force:
            print(f"skip existing: {output_path}")
            continue
        payloads.append(
            {
                "pcap_path": str(pcap_path),
                "output_path": str(output_path),
                "label_map": str(args.label_map),
                "label_csv": summary["output_path"],
                "max_packets_per_flow": args.max_packets_per_flow,
                "max_packets_to_read": args.max_packets_to_read,
                "min_packet_time": None if args.no_time_window else summary["window_start_unix"],
                "max_packet_time": None if args.no_time_window else summary["window_end_unix"],
            }
        )

    built = []
    if payloads:
        with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = [executor.submit(_build_one, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
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
    merged_stats = _merge_outputs(shard_paths, args.merged_path)
    merged_frame = pd.read_parquet(args.merged_path)
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
    time_block_small_stats = None
    if _has_low_support(time_block_stats, args.low_support_min):
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
                "built": built,
                "merged_path": str(args.merged_path),
                "merged_stats": merged_stats,
                "split_dir": str(args.split_dir),
                "split_stats": split_stats,
                "time_ordered_split_dir": str(args.time_ordered_split_dir),
                "time_ordered_stats": time_ordered_stats,
                "time_block_split_dir": str(args.time_block_split_dir),
                "time_block_stats": time_block_stats,
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
