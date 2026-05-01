"""Download and prepare a small CICIDS2017 Friday PCAP smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import pandas as pd

from traffic_bert.config import write_json
from traffic_bert.data.build import BuildConfig, build_processed_dataset
from traffic_bert.data.split import assign_stratified_hash_split, processed_stats, stratified_sample
from traffic_bert.data.validate import validation_summary
from traffic_bert.data.schema import InputView


DEFAULT_REPO_ID = "bencorn/CICIDS2017"
FRIDAY_PCAP = "pcaps/Friday-WorkingHours.pcap"
LABEL_ZIP = "csvs/GeneratedLabelledFlows.zip"


def _local_candidate(raw_dir: Path, repo_filename: str) -> Path:
    nested = raw_dir / repo_filename
    if nested.exists():
        return nested
    return raw_dir / Path(repo_filename).name


def _download_from_huggingface(repo_id: str, filename: str, raw_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for download; install project dependencies with uv"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )


def _find_friday_csvs(label_zip: Path, name_contains: str | None = None) -> list[str]:
    needle = name_contains.lower() if name_contains else None
    with zipfile.ZipFile(label_zip) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and "friday" in Path(name).name.lower()
            and (needle is None or needle in Path(name).name.lower())
        ]
    if not names:
        raise FileNotFoundError(f"no Friday labelled flow CSV found in {label_zip}")
    return sorted(names)


def _extract_and_merge_friday_labels(
    label_zip: Path,
    output_path: Path,
    name_contains: str | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    csv_names = _find_friday_csvs(label_zip, name_contains)
    with zipfile.ZipFile(label_zip) as archive:
        for name in csv_names:
            with archive.open(name) as handle:
                frame = pd.read_csv(handle, encoding="latin1", low_memory=False)
                frame["source_label_file"] = Path(name).name
                frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_path, index=False)
    label_column = next(
        (column for column in merged.columns if column.strip().lower() == "label"),
        None,
    )
    return {
        "label_zip": str(label_zip),
        "csv_files": csv_names,
        "output_path": str(output_path),
        "rows": int(len(merged)),
        "name_contains": name_contains,
        "labels": merged[label_column].value_counts().to_dict()
        if label_column is not None
        else {},
    }


def _write_split_outputs(frame: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frame = frame[frame["split"] == split_name]
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        split_frame["major_label"].value_counts().rename_axis("major_label").reset_index(
            name="rows"
        ).to_csv(output_dir / f"{split_name}.class_distribution.csv", index=False)
        write_json(
            output_dir / f"{split_name}.validate.json",
            validation_summary(split_frame),
        )
    stats = processed_stats(frame)
    write_json(output_dir / "split.stats.json", stats)
    return stats


def _build_smoke_subset(
    input_path: Path,
    output_dir: Path,
    max_per_major: int,
    seed: int,
) -> dict:
    frame = pd.read_parquet(input_path)
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
    return _write_split_outputs(sampled, output_dir)


def _run_smoke_training(output_dir: Path, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv",
            "run",
            "traffic-bert",
            "train",
            "neural-baseline",
            "--train-path",
            str(output_dir / "train.parquet"),
            "--val-path",
            str(output_dir / "val.parquet"),
            "--output-dir",
            str(artifact_dir / "neural_baseline"),
            "--view",
            "payload_only",
            "--model",
            "cnn",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--max-windows",
            "2",
            "--device",
            "auto",
            "--log-path",
            str(artifact_dir / "neural_baseline" / "train.jsonl"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "traffic-bert",
            "eval",
            "neural-baseline",
            "--data-path",
            str(output_dir / "test.parquet"),
            "--checkpoint",
            str(artifact_dir / "neural_baseline" / "neural_baseline.pt"),
            "--view",
            "payload_only",
            "--batch-size",
            "4",
            "--max-windows",
            "2",
            "--device",
            "auto",
            "--output-dir",
            str(artifact_dir / "eval"),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/CICIDS2017"))
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed/cicids2017/friday_payload_only/flows.parquet"),
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("data/processed/cicids2017/smoke"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/cicids2017_smoke"),
    )
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument(
        "--label-file-contains",
        help="Optional Friday labelled CSV filename filter, e.g. Morning, PortScan, or DDos.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--max-packets-to-read", type=int, default=250_000)
    parser.add_argument("--max-packets-to-skip", type=int, default=0)
    parser.add_argument(
        "--start-time",
        help="Optional packet window start as ISO timestamp or Unix seconds.",
    )
    parser.add_argument(
        "--end-time",
        help="Optional packet window end as ISO timestamp or Unix seconds.",
    )
    parser.add_argument("--max-packets-per-flow", type=int, default=16)
    parser.add_argument("--max-per-major", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-train", action="store_true")
    return parser.parse_args()


def _parse_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    pcap_path = _local_candidate(raw_dir, FRIDAY_PCAP)
    zip_path = _local_candidate(raw_dir, LABEL_ZIP)

    if not args.skip_download:
        zip_path = _download_from_huggingface(args.repo_id, LABEL_ZIP, raw_dir)
        pcap_path = _download_from_huggingface(args.repo_id, FRIDAY_PCAP, raw_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"missing labelled flow zip: {zip_path}")

    label_suffix = (
        args.label_file_contains.lower().replace(" ", "_")
        if args.label_file_contains
        else "all"
    )
    label_csv = raw_dir / "labels" / f"friday_labelled_flows_{label_suffix}.csv"
    if args.force_labels or not label_csv.exists():
        label_stats = _extract_and_merge_friday_labels(
            zip_path,
            label_csv,
            name_contains=args.label_file_contains,
        )
        write_json(label_csv.with_suffix(".summary.json"), label_stats)

    if not pcap_path.exists():
        raise FileNotFoundError(f"missing Friday PCAP: {pcap_path}")

    if args.force_build or not args.processed_path.exists():
        min_packet_time = _parse_timestamp(args.start_time)
        max_packet_time = _parse_timestamp(args.end_time)
        stats = build_processed_dataset(
            BuildConfig(
                input_path=pcap_path,
                output_path=args.processed_path,
                label_map_path=args.label_map,
                source_dataset="cicids2017-friday",
                split="train",
                label_source="cic_csv",
                label_csv_path=label_csv,
                drop_unmatched_labels=True,
                views=(InputView.PAYLOAD_ONLY,),
                keep_empty_payload=False,
                max_packets_per_flow=args.max_packets_per_flow,
                max_packets_to_read=args.max_packets_to_read,
                max_packets_to_skip=args.max_packets_to_skip,
                min_packet_time=min_packet_time,
                max_packet_time=max_packet_time,
            )
        )
        write_json(
            args.processed_path.with_suffix(".build.json"),
            {
                **stats,
                "max_packets_to_read": args.max_packets_to_read,
                "max_packets_to_skip": args.max_packets_to_skip,
                "max_packets_per_flow": args.max_packets_per_flow,
                "start_time": args.start_time,
                "end_time": args.end_time,
            },
        )

    smoke_stats = _build_smoke_subset(
        input_path=args.processed_path,
        output_dir=args.smoke_dir,
        max_per_major=args.max_per_major,
        seed=args.seed,
    )
    write_json(args.artifact_dir / "smoke_data_stats.json", smoke_stats)

    if args.run_train:
        _run_smoke_training(args.smoke_dir, args.artifact_dir)

    print(json.dumps(smoke_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
