"""Fail-fast checks for datasets that are eligible for formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


CICIDS_PCAPS = (
    "Monday-WorkingHours.pcap",
    "Tuesday-WorkingHours.pcap",
    "Wednesday-workingHours.pcap",
    "Thursday-WorkingHours.pcap",
    "Friday-WorkingHours.pcap",
)

CICIDS_SHARDS = (
    "monday_benign",
    "tuesday_patator",
    "wednesday_dos",
    "thursday_web",
    "thursday_infiltration",
    "friday_bot",
    "friday_portscan",
    "friday_ddos",
)

REQUIRED_CICIDS_TIMESTAMP_POLICY_VERSION = 2
REQUIRED_CICIDS_WINDOW_SCOPE = "pcap"
REQUIRED_CONNECTION_TYPE_VERSION = 1
REQUIRED_TCP_CLOSE_POLICY_VERSION = 3
REQUIRED_CLOSE_ON_TCP_FLAGS = False
REQUIRED_CSV_TIME_OFFSET_HOURS = 3.0

FORBIDDEN_FORMAL_PATH_PARTS = {
    "smoke",
    "friday_payload_only",
    "payload_byte",
    "split_label_stratified",
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _path_parts(path: Path) -> set[str]:
    return {part.lower() for part in path.parts}


def _check_file(path: Path, description: str, errors: list[str]) -> None:
    if not _is_nonempty_file(path):
        errors.append(f"missing or empty {description}: {path}")


def _check_processed_parquet_schema(
    path: Path,
    description: str,
    required_columns: set[str],
    errors: list[str],
) -> None:
    if not _is_nonempty_file(path):
        return
    try:
        schema_names = set(pq.read_schema(path).names)
    except Exception as exc:  # pragma: no cover - defensive around parquet engines.
        errors.append(f"could not read {description} parquet schema {path}: {exc}")
        return
    missing = sorted(required_columns - schema_names)
    if missing:
        errors.append(f"{description} missing required columns {missing}: {path}")


def _check_required_view(path: Path, description: str, required_view: str | None, errors: list[str]) -> None:
    if not required_view or not _is_nonempty_file(path):
        return
    try:
        views = set(pd.read_parquet(path, columns=["view"])["view"].dropna().astype(str))
    except Exception as exc:  # pragma: no cover - defensive around parquet engines.
        errors.append(f"could not read {description} view column {path}: {exc}")
        return
    required_views = {item.strip() for item in required_view.split(",") if item.strip()}
    missing = sorted(required_views - views)
    if missing:
        errors.append(f"{description} missing required view values {missing}; found {sorted(views)}: {path}")


def _check_split_paths(args: argparse.Namespace, errors: list[str]) -> None:
    required_columns = {
        "flow_id",
        "source_file",
        "source_label",
        "major_label",
        "view",
        "protocol",
        "connection_type",
        "initiator_endpoint",
        "responder_endpoint",
        "label_match_mode",
        "label_time_delta_seconds",
        "has_payload",
        "observed_packet_count",
        "was_packet_truncated",
        "packet_directions",
        "packet_lengths",
        "bytes",
    }
    for split, path in [
        ("train", args.train_path),
        ("val", args.val_path),
        ("test", args.test_path),
    ]:
        _check_file(path, f"{split} split parquet", errors)
        _check_processed_parquet_schema(path, f"{split} split", required_columns, errors)
        _check_required_view(path, f"{split} split", getattr(args, "required_view", None), errors)
        forbidden = sorted(_path_parts(path) & FORBIDDEN_FORMAL_PATH_PARTS)
        if forbidden:
            errors.append(
                f"{split} split path is not formal-eligible ({', '.join(forbidden)}): {path}"
            )
    if args.dataset == "cicids2017":
        expected_parent = getattr(
            args,
            "required_split_parent",
            Path("data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose"),
        )
        for split, path in [
            ("train", args.train_path),
            ("val", args.val_path),
            ("test", args.test_path),
        ]:
            normalized_parent = path.parent.as_posix().replace("\\", "/")
            if not normalized_parent.endswith(expected_parent.as_posix()):
                errors.append(
                    f"{split} split must come from {expected_parent.as_posix()} for "
                    f"formal CICIDS2017 training: {path}"
                )


def _check_cicids_raw(args: argparse.Namespace, errors: list[str]) -> None:
    for filename in CICIDS_PCAPS:
        _check_file(args.raw_dir / "pcaps" / filename, f"CICIDS2017 PCAP {filename}", errors)
    _check_file(args.label_zip, "CICIDS2017 label zip", errors)


def _check_cicids_build_metadata(args: argparse.Namespace, errors: list[str]) -> None:
    _check_file(args.merged_path, "CICIDS2017 merged flows_all parquet", errors)
    _check_processed_parquet_schema(
        args.merged_path,
        "CICIDS2017 merged flows_all",
        {
            "flow_id",
            "source_file",
            "source_label",
            "major_label",
            "view",
            "protocol",
            "connection_type",
            "initiator_endpoint",
            "responder_endpoint",
            "label_match_mode",
            "label_time_delta_seconds",
            "has_payload",
            "observed_packet_count",
            "was_packet_truncated",
            "packet_directions",
            "packet_lengths",
            "bytes",
        },
        errors,
    )
    _check_required_view(
        args.merged_path,
        "CICIDS2017 merged flows_all",
        getattr(args, "required_view", None),
        errors,
    )
    for slug in CICIDS_SHARDS:
        path = args.processed_dir / f"flows_{slug}.build.json"
        _check_file(path, f"CICIDS2017 build metadata {slug}", errors)
        if not path.exists():
            continue
        try:
            data = _read_json(path)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid build metadata JSON {path}: {exc}")
            continue
        if "flow_timeout_seconds" not in data:
            errors.append(f"stale build metadata missing flow_timeout_seconds: {path}")
        else:
            actual_timeout = float(data["flow_timeout_seconds"])
            if actual_timeout != float(args.required_flow_timeout_seconds):
                errors.append(
                    f"unexpected flow_timeout_seconds in {path}: "
                    f"{actual_timeout:g} != {args.required_flow_timeout_seconds:g}"
                )
        if data.get("cicids_timestamp_policy_version") != args.required_timestamp_policy_version:
            errors.append(
                f"stale build metadata timestamp policy in {path}: "
                f"{data.get('cicids_timestamp_policy_version')} != "
                f"{args.required_timestamp_policy_version}"
            )
        if data.get("connection_type_version") != args.required_connection_type_version:
            errors.append(
                f"stale build metadata connection type version in {path}: "
                f"{data.get('connection_type_version')} != "
                f"{args.required_connection_type_version}"
            )
        if data.get("tcp_close_policy_version") != args.required_tcp_close_policy_version:
            errors.append(
                f"stale build metadata TCP close policy version in {path}: "
                f"{data.get('tcp_close_policy_version')} != "
                f"{args.required_tcp_close_policy_version}"
            )
        if data.get("close_on_tcp_flags") is not args.required_close_on_tcp_flags:
            errors.append(
                f"unexpected close_on_tcp_flags in {path}: "
                f"{data.get('close_on_tcp_flags')} != {args.required_close_on_tcp_flags}"
            )
        actual_offset = data.get("csv_time_offset_hours")
        if actual_offset != args.required_csv_time_offset_hours:
            errors.append(
                f"unexpected csv_time_offset_hours in {path}: "
                f"{actual_offset} != {args.required_csv_time_offset_hours}"
            )
        if data.get("window_scope") != args.required_window_scope:
            errors.append(
                f"unexpected CICIDS window_scope in {path}: "
                f"{data.get('window_scope')!r} != {args.required_window_scope!r}"
            )
        actual_delta = data.get("cic_label_max_time_delta_seconds")
        if actual_delta != args.required_cic_label_max_time_delta_seconds:
            errors.append(
                f"unexpected cic_label_max_time_delta_seconds in {path}: "
                f"{actual_delta} != {args.required_cic_label_max_time_delta_seconds}"
            )
        required_view = getattr(args, "required_view", None)
        if required_view is not None:
            views = data.get("views") or data.get("build_config", {}).get("views") or []
            required_views = [item.strip() for item in required_view.split(",") if item.strip()]
            missing_views = [view for view in required_views if view not in views]
            if missing_views:
                errors.append(f"missing required views in {path}: {missing_views}; found {views}")
        required_keep = getattr(args, "required_keep_empty_payload", None)
        if required_keep is not None:
            actual_keep = data.get("keep_empty_payload")
            if actual_keep is None:
                actual_keep = data.get("build_config", {}).get("keep_empty_payload")
            if isinstance(actual_keep, str):
                actual_keep_bool = actual_keep.strip().lower() in {"1", "true", "yes", "y"}
            else:
                actual_keep_bool = bool(actual_keep)
            if actual_keep_bool != required_keep:
                errors.append(
                    f"unexpected keep_empty_payload in {path}: "
                    f"{actual_keep} != {required_keep}"
                )


def _check_json_eligible(path: Path, description: str, errors: list[str]) -> None:
    _check_file(path, description, errors)
    if not path.exists():
        return
    try:
        data = _read_json(path)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid {description} JSON {path}: {exc}")
        return
    if data.get("formal_eligible") is not True:
        errors.append(f"{description} is not formal_eligible=true: {path}")
    if data.get("failures"):
        errors.append(f"{description} contains failures: {path}")
    if data.get("blocking_warnings"):
        errors.append(f"{description} contains blocking_warnings: {path}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    _check_split_paths(args, errors)

    if args.dataset == "cicids2017":
        _check_cicids_raw(args, errors)
        _check_cicids_build_metadata(args, errors)
        _check_json_eligible(args.attack_coverage_path, "CICIDS2017 attack coverage", errors)

    if args.audit_path is not None:
        _check_json_eligible(args.audit_path, "split audit", errors)
    elif args.dataset == "cicids2017" and not args.allow_missing_audit:
        errors.append(
            "CICIDS2017 formal verification requires --audit-path; "
            "use --allow-missing-audit only for a pre-audit schema check"
        )

    return {
        "dataset": args.dataset,
        "formal_eligible": not errors,
        "errors": errors,
        "checked": {
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "test_path": str(args.test_path),
            "audit_path": None if args.audit_path is None else str(args.audit_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cicids2017"], default="cicids2017")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(
            "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet"
        ),
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path(
            "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet"
        ),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(
            "data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet"
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=Path(
            "artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json"
        ),
    )
    parser.add_argument(
        "--allow-missing-audit",
        action="store_true",
        help="Allow a CICIDS schema/coverage preflight before the split audit has been written.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/CICIDS2017"))
    parser.add_argument(
        "--label-zip",
        type=Path,
        default=Path("data/raw/CICIDS2017/csvs/GeneratedLabelledFlows.zip"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose"),
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_all.parquet"),
    )
    parser.add_argument(
        "--attack-coverage-path",
        type=Path,
        default=Path("artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json"),
    )
    parser.add_argument("--required-flow-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--required-connection-type-version",
        type=int,
        default=REQUIRED_CONNECTION_TYPE_VERSION,
    )
    parser.add_argument(
        "--required-tcp-close-policy-version",
        type=int,
        default=REQUIRED_TCP_CLOSE_POLICY_VERSION,
    )
    parser.add_argument(
        "--required-close-on-tcp-flags",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"},
        default=REQUIRED_CLOSE_ON_TCP_FLAGS,
    )
    parser.add_argument(
        "--required-csv-time-offset-hours",
        type=float,
        default=REQUIRED_CSV_TIME_OFFSET_HOURS,
    )
    parser.add_argument(
        "--required-split-parent",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose"),
    )
    parser.add_argument("--required-view", default=None)
    parser.add_argument(
        "--required-keep-empty-payload",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"},
        default=None,
    )
    parser.add_argument(
        "--required-timestamp-policy-version",
        type=int,
        default=REQUIRED_CICIDS_TIMESTAMP_POLICY_VERSION,
    )
    parser.add_argument(
        "--required-window-scope",
        choices=["pcap", "label", "attack"],
        default=REQUIRED_CICIDS_WINDOW_SCOPE,
    )
    parser.add_argument("--required-cic-label-max-time-delta-seconds", type=float, default=900.0)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result["formal_eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
