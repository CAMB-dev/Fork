"""Canonical dataset preparation entrypoint.

This script owns the formal preprocessing defaults. Shell and PowerShell files
should be thin wrappers around this file instead of carrying their own path and
gate logic.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_bert.config import write_json
from traffic_bert.data.build import BuildConfig, build_processed_dataset
from traffic_bert.data.schema import InputView, collect_pcap_files
from traffic_bert.data.split import (
    assign_file_time_split,
    assign_stratified_hash_split,
    processed_stats,
    split_run_stats,
    stratified_sample,
)
from traffic_bert.data.validate import validation_summary


CICIDS_PROCESSED_DIR = Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose")
CICIDS_MERGED_PATH = CICIDS_PROCESSED_DIR / "flows_all.parquet"
CICIDS_RANDOM_SPLIT_DIR = Path("data/processed/cicids2017/all_masked_header_split_label_stratified_cap32_notcpclose")
CICIDS_TIME_ORDERED_SPLIT_DIR = Path("data/processed/cicids2017/all_masked_header_split_time_ordered_cap32_notcpclose")
CICIDS_TIME_BLOCK_SPLIT_DIR = Path("data/processed/cicids2017/all_masked_header_split_time_block_cap32_notcpclose")
CICIDS_SUBMODE_STRATIFIED_SPLIT_DIR = Path(
    "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose"
)
CICIDS_TIME_BLOCK_SMALL_SPLIT_DIR = Path(
    "data/processed/cicids2017/all_masked_header_split_time_block_128_cap32_notcpclose"
)
CICIDS_COVERAGE_DIR = Path("artifacts/cicids2017_coverage_masked_header_cap32_notcpclose")
CICIDS_AUDIT_DIR = Path("artifacts/cicids2017_all_masked_header_cap32_notcpclose")

USTC_SOURCE_ROOT = Path("data/raw/USTC-TFC2016/extracted/USTC-TFC2016-master")
USTC_FILES_DIR = Path("data/processed/ustc_tfc2016/files")
USTC_LOG_DIR = Path("data/processed/ustc_tfc2016/logs")
USTC_MERGED_PATH = Path("data/processed/ustc_tfc2016/merged/all.parquet")
USTC_LABEL_SPLIT_DIR = Path("data/processed/ustc_tfc2016/split_label_stratified")
USTC_SOURCE_SPLIT_DIR = Path("data/processed/ustc_tfc2016/split_source_file")
USTC_MAJOR_SOURCE_SPLIT_DIR = Path("data/processed/ustc_tfc2016/split_source_file_major_balanced")
USTC_AUDIT_DIR = Path("artifacts/ustc_tfc2016")


def _run(cmd: Sequence[str | Path], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(item) for item in cmd)
    print(f"\n==== {printable} ====")
    subprocess.run([str(item) for item in cmd], check=True, env=env)


def _run_python(args: Sequence[str | Path]) -> None:
    _run([sys.executable, *args])


def _run_download(dataset: str) -> None:
    if shutil.which("bash") is None:
        raise RuntimeError("bash is required for scripts/download_datasets.sh")
    env = dict(**__import__("os").environ, DATASET=dataset)
    _run(["bash", "scripts/download_datasets.sh"], env=env)


def _validate_split(split_dir: Path) -> None:
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing split parquet: {path}")
        frame = pd.read_parquet(path)
        write_json(path.with_suffix(".validate.json"), validation_summary(frame))


def _audit_split(
    split_dir: Path,
    output_path: Path,
    *,
    low_support_labels: Sequence[str] = (),
    low_support_min: int = 2,
    fail_on_warnings: bool = False,
    ignore_source_file_overlap: bool = False,
) -> None:
    args: list[str | Path] = [
        "scripts/audit_processed_split.py",
        "--train-path",
        split_dir / "train.parquet",
        "--val-path",
        split_dir / "val.parquet",
        "--test-path",
        split_dir / "test.parquet",
        "--low-support-min",
        str(low_support_min),
        "--output-path",
        output_path,
    ]
    for label in low_support_labels:
        args.extend(["--low-support-label", label])
    if fail_on_warnings:
        args.append("--fail-on-warnings")
    if ignore_source_file_overlap:
        args.append("--ignore-source-file-overlap-for-eligibility")
    _run_python(args)


def prepare_cicids(args: argparse.Namespace) -> None:
    if args.skip_audit:
        raise ValueError("formal CICIDS preprocessing cannot use --skip-audit")
    if args.cicids_window_scope != "pcap":
        raise ValueError("formal CICIDS preprocessing must use --cicids-window-scope pcap")
    if float(args.csv_time_offset_hours) != 3.0:
        raise ValueError(
            "formal CICIDS preprocessing expects --csv-time-offset-hours 3.0; "
            "run alignment diagnostics before changing it"
        )
    if args.cicids_drop_empty_payload:
        raise ValueError("formal CICIDS preprocessing must keep empty-payload flows")
    if "masked_header_packet" not in {
        item.strip() for item in args.cicids_views.split(",") if item.strip()
    }:
        raise ValueError("formal CICIDS preprocessing requires masked_header_packet")

    if not args.skip_download:
        _run_download("cicids2017-all")

    build_args: list[str | Path] = [
        "scripts/prepare_cicids2017_all_parallel.py",
        "--output-dir",
        args.cicids_processed_dir,
        "--merged-path",
        args.cicids_merged_path,
        "--split-dir",
        args.cicids_random_split_dir,
        "--time-ordered-split-dir",
        args.cicids_time_ordered_split_dir,
        "--time-block-split-dir",
        args.cicids_time_block_split_dir,
        "--submode-stratified-split-dir",
        args.cicids_submode_stratified_split_dir,
        "--time-block-small-split-dir",
        args.cicids_time_block_small_split_dir,
        "--audit-dir",
        args.cicids_coverage_dir,
        "--max-workers",
        str(args.max_workers),
        "--max-packets-per-flow",
        str(args.max_packets_per_flow),
        "--flow-timeout-seconds",
        str(args.flow_timeout_seconds),
        "--close-on-tcp-flags" if args.close_on_tcp_flags else "--no-close-on-tcp-flags",
        "--cic-label-max-time-delta-seconds",
        str(args.cicids_label_max_time_delta_seconds),
        "--views",
        args.cicids_views,
        "--padding-minutes",
        str(args.padding_minutes),
        "--csv-time-offset-hours",
        str(args.csv_time_offset_hours),
        "--window-scope",
        args.cicids_window_scope,
        "--max-per-major",
        str(args.cicids_max_per_major),
        "--min-attack-flows-per-label",
        str(args.cicids_min_attack_flows_per_label),
        "--min-attack-match-ratio",
        str(args.cicids_min_attack_match_ratio),
        "--time-block-size",
        str(args.time_block_size),
        "--low-support-min",
        str(args.low_support_min),
    ]
    if not args.include_nonformal_splits:
        build_args.append("--formal-only-splits")
    if args.max_packets_to_read is not None:
        build_args.extend(["--max-packets-to-read", str(args.max_packets_to_read)])
    if args.force:
        build_args.append("--force")
    if args.cicids_drop_empty_payload:
        build_args.append("--drop-empty-payload")
    _run_python(build_args)

    if not args.skip_validate:
        split_dirs = [args.cicids_submode_stratified_split_dir]
        if args.include_nonformal_splits:
            split_dirs.extend(
                [
                    args.cicids_random_split_dir,
                    args.cicids_time_ordered_split_dir,
                    args.cicids_time_block_split_dir,
                ]
            )
        for split_dir in split_dirs:
            _validate_split(split_dir)
        if args.include_nonformal_splits and args.cicids_time_block_small_split_dir.exists():
            _validate_split(args.cicids_time_block_small_split_dir)

    if not args.skip_audit:
        args.cicids_all_audit_dir.mkdir(parents=True, exist_ok=True)
        if args.audit_nonformal_splits:
            _audit_split(
                args.cicids_random_split_dir,
                args.cicids_all_audit_dir / "random_flow_split_audit.json",
                low_support_min=args.low_support_min,
            )
            _audit_split(
                args.cicids_time_ordered_split_dir,
                args.cicids_all_audit_dir / "time_ordered_split_audit.json",
                low_support_min=args.low_support_min,
            )
        _audit_split(
            args.cicids_submode_stratified_split_dir,
            args.cicids_all_audit_dir / "submode_stratified_group_split_audit.json",
            low_support_min=args.low_support_min,
            fail_on_warnings=True,
            ignore_source_file_overlap=True,
        )
        if args.audit_nonformal_splits:
            _audit_split(
                args.cicids_time_block_split_dir,
                args.cicids_all_audit_dir / "time_block_split_audit.json",
                low_support_min=args.low_support_min,
                ignore_source_file_overlap=True,
            )
        if args.audit_nonformal_splits and args.cicids_time_block_small_split_dir.exists():
            _audit_split(
                args.cicids_time_block_small_split_dir,
                args.cicids_all_audit_dir / "time_block_128_split_audit.json",
                low_support_min=args.low_support_min,
            )

    gate_args: list[str | Path] = [
        "scripts/verify_formal_dataset.py",
        "--dataset",
        "cicids2017",
        "--train-path",
        args.cicids_submode_stratified_split_dir / "train.parquet",
        "--val-path",
        args.cicids_submode_stratified_split_dir / "val.parquet",
        "--test-path",
        args.cicids_submode_stratified_split_dir / "test.parquet",
        "--processed-dir",
        args.cicids_processed_dir,
        "--merged-path",
        args.cicids_merged_path,
        "--attack-coverage-path",
        args.cicids_coverage_dir / "attack_coverage.json",
        "--required-split-parent",
        args.cicids_submode_stratified_split_dir,
        "--required-view",
        "masked_header_packet",
        "--required-keep-empty-payload",
        "true",
        "--required-flow-timeout-seconds",
        str(args.flow_timeout_seconds),
        "--required-window-scope",
        "pcap",
        "--required-csv-time-offset-hours",
        "3.0",
        "--required-cic-label-max-time-delta-seconds",
        str(args.cicids_label_max_time_delta_seconds),
        "--required-close-on-tcp-flags",
        "true" if args.close_on_tcp_flags else "false",
        "--output-path",
        args.cicids_all_audit_dir / "formal_dataset_gate.json",
    ]
    if not args.skip_audit:
        gate_args.extend(
            [
                "--audit-path",
                args.cicids_all_audit_dir / "submode_stratified_group_split_audit.json",
            ]
        )
    _run_python(gate_args)


def _ustc_output_path(pcap: Path, source_root: Path, output_dir: Path) -> Path:
    kind = "Benign" if "Benign" in pcap.relative_to(source_root).parts else "Malware"
    safe_name = "".join(ch if ch not in '\\/:*?"<>| ' else "_" for ch in pcap.stem)
    return output_dir / f"{kind}_{safe_name}.parquet"


def _is_fresh_ustc(path: Path, *, max_packets_per_flow: int, flow_timeout_seconds: float) -> bool:
    build_path = path.with_suffix(".build.json")
    if not path.exists() or not build_path.exists():
        return False
    try:
        data = json.loads(build_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("max_packets_per_flow") == max_packets_per_flow
        and float(data.get("flow_timeout_seconds", -1)) == float(flow_timeout_seconds)
        and data.get("views") == ["payload_only"]
        and data.get("keep_empty_payload") is False
        and data.get("connection_type_version") == 1
        and data.get("tcp_close_policy_version") == 3
        and data.get("close_on_tcp_flags") is False
    )


def _build_ustc_one(payload: dict) -> dict:
    pcap = Path(payload["pcap"])
    output_path = Path(payload["output_path"])
    source_root = Path(payload["source_root"])
    kind = "Benign" if "Benign" in pcap.relative_to(source_root).parts else "Malware"
    config = BuildConfig(
        input_path=pcap,
        output_path=output_path,
        label_map_path=Path(payload["label_map"]),
        source_dataset="ustc-tfc2016",
        split="train",
        label_source="static" if kind == "Benign" else "filename",
        static_label="BENIGN" if kind == "Benign" else None,
        views=(InputView.PAYLOAD_ONLY,),
        keep_empty_payload=False,
        max_packets_per_flow=payload["max_packets_per_flow"],
        flow_timeout_seconds=payload["flow_timeout_seconds"],
    )
    stats = build_processed_dataset(config)
    write_json(
        output_path.with_suffix(".build.json"),
        {
            **stats,
            "input_path": str(pcap),
            "max_packets_per_flow": payload["max_packets_per_flow"],
            "flow_timeout_seconds": payload["flow_timeout_seconds"],
            "views": ["payload_only"],
            "keep_empty_payload": False,
            "connection_type_version": 1,
            "tcp_close_policy_version": 3,
            "close_on_tcp_flags": False,
        },
    )
    return {"pcap": str(pcap), "output_path": str(output_path), "rows": stats.get("rows", 0)}


def _merge_parquet(paths: Sequence[Path], output_path: Path) -> None:
    if not paths:
        raise FileNotFoundError("no parquet files to merge")
    tables = [pq.read_table(path) for path in paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables, promote_options="default"), output_path, compression="zstd")


def _write_split_outputs(frame: pd.DataFrame, output_dir: Path, stats: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split]
        split_frame.to_parquet(output_dir / f"{split}.parquet", index=False)
        write_json(output_dir / f"{split}.validate.json", validation_summary(split_frame))
    write_json(output_dir / "split.stats.json", stats)


def _create_ustc_splits(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(args.ustc_merged_path)

    sampled = stratified_sample(
        frame,
        stratify_column="major_label",
        max_per_class=1_000_000_000,
        seed=args.seed,
    )
    sampled = assign_stratified_hash_split(
        sampled,
        group_column="flow_id",
        stratify_column="source_label",
    )
    _write_split_outputs(
        sampled,
        args.ustc_label_split_dir,
        split_run_stats(
            input_frame=frame,
            output_frame=sampled,
            split_method="random_flow_hash",
            max_per_class=1_000_000_000,
            stratify_column="major_label",
            split_stratify_column="source_label",
            group_column="flow_id",
            seed=args.seed,
            train_ratio=0.7,
            val_ratio=0.15,
        ),
    )

    source_split = assign_file_time_split(frame, group_column="source_file")
    _write_split_outputs(source_split, args.ustc_source_split_dir, processed_stats(source_split))

    major_split = assign_stratified_hash_split(
        frame,
        group_column="source_file",
        stratify_column="major_label",
    )
    _write_split_outputs(major_split, args.ustc_major_source_split_dir, processed_stats(major_split))


def prepare_ustc(args: argparse.Namespace) -> None:
    if not args.skip_download and not args.ustc_source_root.exists():
        _run_download("ustc_tfc2016")
    if not args.ustc_source_root.exists():
        raise FileNotFoundError(f"USTC source root not found: {args.ustc_source_root}")

    args.ustc_files_dir.mkdir(parents=True, exist_ok=True)
    args.ustc_log_dir.mkdir(parents=True, exist_ok=True)
    pcaps = collect_pcap_files(args.ustc_source_root)
    if not pcaps:
        raise FileNotFoundError(f"no USTC PCAP files found under {args.ustc_source_root}")

    payloads = []
    outputs = []
    for pcap in pcaps:
        output_path = _ustc_output_path(pcap, args.ustc_source_root, args.ustc_files_dir)
        outputs.append(output_path)
        if not args.force and _is_fresh_ustc(
            output_path,
            max_packets_per_flow=args.max_packets_per_flow,
            flow_timeout_seconds=args.flow_timeout_seconds,
        ):
            print(f"skip fresh USTC shard: {output_path}")
            continue
        payloads.append(
            {
                "pcap": str(pcap),
                "output_path": str(output_path),
                "source_root": str(args.ustc_source_root),
                "label_map": str(args.label_map),
                "max_packets_per_flow": args.max_packets_per_flow,
                "flow_timeout_seconds": args.flow_timeout_seconds,
            }
        )

    if payloads:
        print(
            "Parallel USTC build: "
            f"{len(payloads)} stale/missing PCAPs, workers={max(1, args.ustc_workers)}"
        )
        with ProcessPoolExecutor(max_workers=max(1, args.ustc_workers)) as executor:
            futures = [executor.submit(_build_ustc_one, payload) for payload in payloads]
            for future in as_completed(futures):
                print(json.dumps(future.result(), ensure_ascii=False))

    existing_outputs = [path for path in outputs if path.exists()]
    _merge_parquet(existing_outputs, args.ustc_merged_path)
    _create_ustc_splits(args)

    if not args.skip_audit:
        args.ustc_audit_dir.mkdir(parents=True, exist_ok=True)
        for split_dir, output_name in (
            (args.ustc_label_split_dir, "split_label_stratified_audit.json"),
            (args.ustc_source_split_dir, "split_source_file_audit.json"),
            (args.ustc_major_source_split_dir, "split_source_file_major_balanced_audit.json"),
        ):
            _audit_split(
                split_dir,
                args.ustc_audit_dir / output_name,
                low_support_labels=("botnet_malware",),
                low_support_min=args.low_support_min,
            )


def prepare_friday_smoke(args: argparse.Namespace) -> None:
    if not args.skip_download:
        _run_download("cicids2017-friday-smoke")
    smoke_args: list[str | Path] = [
        "scripts/cicids2017_friday_smoke.py",
        "--max-packets-to-read",
        str(args.max_packets_to_read or 250_000),
        "--max-packets-to-skip",
        str(args.max_packets_to_skip),
        "--max-packets-per-flow",
        str(args.max_packets_per_flow),
        "--max-per-major",
        str(args.max_per_major),
    ]
    if args.skip_download:
        smoke_args.append("--skip-download")
    if args.force:
        smoke_args.append("--force-build")
    if args.run_smoke_train:
        smoke_args.append("--run-train")
    _run_python(smoke_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cicids2017",
        choices=[
            "cicids2017",
            "cicids2017-all",
            "ustc",
            "ustc_tfc2016",
            "all",
            "cicids2017-friday-smoke",
        ],
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--audit-nonformal-splits",
        action="store_true",
        help=(
            "Also audit random-flow/time-ordered/time-block CICIDS sanity splits; "
            "formal gate only needs submode-stratified group."
        ),
    )
    parser.add_argument(
        "--include-nonformal-splits",
        action="store_true",
        help="Also write random-flow/time-ordered/time-block CICIDS sanity splits.",
    )
    parser.add_argument("--run-smoke-train", action="store_true")
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--ustc-workers", type=int, default=4)
    parser.add_argument("--max-packets-per-flow", type=int, default=32)
    parser.add_argument("--max-packets-to-read", type=int, default=None)
    parser.add_argument("--max-packets-to-skip", type=int, default=0)
    parser.add_argument("--flow-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--close-on-tcp-flags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Split TCP flows at FIN/RST boundaries. Formal CICIDS keeps this disabled "
            "to avoid training on detached TCP close/reset fragments."
        ),
    )
    parser.add_argument("--low-support-min", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-major", type=int, default=2_000)

    parser.add_argument("--cicids-processed-dir", type=Path, default=CICIDS_PROCESSED_DIR)
    parser.add_argument("--cicids-merged-path", type=Path, default=CICIDS_MERGED_PATH)
    parser.add_argument("--cicids-random-split-dir", type=Path, default=CICIDS_RANDOM_SPLIT_DIR)
    parser.add_argument(
        "--cicids-time-ordered-split-dir",
        type=Path,
        default=CICIDS_TIME_ORDERED_SPLIT_DIR,
    )
    parser.add_argument("--cicids-time-block-split-dir", type=Path, default=CICIDS_TIME_BLOCK_SPLIT_DIR)
    parser.add_argument(
        "--cicids-submode-stratified-split-dir",
        type=Path,
        default=CICIDS_SUBMODE_STRATIFIED_SPLIT_DIR,
    )
    parser.add_argument(
        "--cicids-time-block-small-split-dir",
        type=Path,
        default=CICIDS_TIME_BLOCK_SMALL_SPLIT_DIR,
    )
    parser.add_argument("--cicids-coverage-dir", type=Path, default=CICIDS_COVERAGE_DIR)
    parser.add_argument("--cicids-all-audit-dir", type=Path, default=CICIDS_AUDIT_DIR)
    parser.add_argument("--cicids-views", default="masked_header_packet")
    parser.add_argument("--cicids-drop-empty-payload", action="store_true")
    parser.add_argument("--cicids-window-scope", choices=["pcap", "label", "attack"], default="pcap")
    parser.add_argument("--cicids-label-max-time-delta-seconds", type=float, default=900.0)
    parser.add_argument("--cicids-min-attack-flows-per-label", type=int, default=100)
    parser.add_argument("--cicids-min-attack-match-ratio", type=float, default=0.05)
    parser.add_argument(
        "--cicids-max-per-major",
        type=int,
        default=0,
        help=(
            "Maximum rows per CICIDS major class when writing formal splits. "
            "Use 0 for no cap; this is the formal full-data default."
        ),
    )
    parser.add_argument("--padding-minutes", type=int, default=20)
    parser.add_argument("--csv-time-offset-hours", type=float, default=3.0)
    parser.add_argument("--time-block-size", type=int, default=512)

    parser.add_argument("--ustc-source-root", type=Path, default=USTC_SOURCE_ROOT)
    parser.add_argument("--ustc-files-dir", type=Path, default=USTC_FILES_DIR)
    parser.add_argument("--ustc-log-dir", type=Path, default=USTC_LOG_DIR)
    parser.add_argument("--ustc-merged-path", type=Path, default=USTC_MERGED_PATH)
    parser.add_argument("--ustc-label-split-dir", type=Path, default=USTC_LABEL_SPLIT_DIR)
    parser.add_argument("--ustc-source-split-dir", type=Path, default=USTC_SOURCE_SPLIT_DIR)
    parser.add_argument("--ustc-major-source-split-dir", type=Path, default=USTC_MAJOR_SOURCE_SPLIT_DIR)
    parser.add_argument("--ustc-audit-dir", type=Path, default=USTC_AUDIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"cicids2017", "cicids2017-all"}:
        prepare_cicids(args)
    elif args.dataset in {"ustc", "ustc_tfc2016"}:
        prepare_ustc(args)
    elif args.dataset == "all":
        prepare_cicids(args)
        prepare_ustc(args)
    elif args.dataset == "cicids2017-friday-smoke":
        prepare_friday_smoke(args)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
