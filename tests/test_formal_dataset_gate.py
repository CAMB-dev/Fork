from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_formal_dataset.py"
SPEC = importlib.util.spec_from_file_location("verify_formal_dataset", SCRIPT_PATH)
assert SPEC is not None
gate_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate_module)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "flow_id": ["f1"],
            "source_file": ["Friday-WorkingHours.pcap"],
            "source_label": ["BENIGN"],
            "major_label": ["benign"],
            "view": ["masked_header_packet"],
            "protocol": ["tcp"],
            "connection_type": ["tcp_payload"],
            "initiator_endpoint": ["10.0.0.1:1234"],
            "responder_endpoint": ["10.0.0.2:80"],
            "label_match_mode": ["directional"],
            "label_time_delta_seconds": [0.0],
            "has_payload": [True],
            "observed_packet_count": [1],
            "was_packet_truncated": [False],
            "packet_directions": [["fwd"]],
            "packet_lengths": [[3]],
            "bytes": [b"abc"],
        }
    ).to_parquet(path, index=False)


def _args(tmp_path: Path, split_dir: Path) -> argparse.Namespace:
    raw_dir = tmp_path / "raw" / "CICIDS2017"
    processed_dir = tmp_path / "processed" / "cicids2017" / "all_masked_header_packet"
    for filename in gate_module.CICIDS_PCAPS:
        pcap_path = raw_dir / "pcaps" / filename
        pcap_path.parent.mkdir(parents=True, exist_ok=True)
        pcap_path.write_bytes(b"pcap")
    label_zip = raw_dir / "csvs" / "GeneratedLabelledFlows.zip"
    label_zip.parent.mkdir(parents=True, exist_ok=True)
    label_zip.write_bytes(b"zip")
    for slug in gate_module.CICIDS_SHARDS:
        _write_json(
            processed_dir / f"flows_{slug}.build.json",
            {
                "flow_timeout_seconds": 120.0,
                "cicids_timestamp_policy_version": 2,
                "window_scope": "pcap",
                "cic_label_max_time_delta_seconds": 900.0,
                "views": ["masked_header_packet"],
                "keep_empty_payload": True,
                "connection_type_version": 1,
                "tcp_close_policy_version": 3,
                "close_on_tcp_flags": False,
                "csv_time_offset_hours": 3.0,
            },
        )
    _write_parquet(processed_dir / "flows_all.parquet")
    for split in ["train", "val", "test"]:
        _write_parquet(split_dir / f"{split}.parquet")
    attack_coverage_path = tmp_path / "artifacts" / "coverage" / "attack_coverage.json"
    audit_path = tmp_path / "artifacts" / "audit.json"
    _write_json(attack_coverage_path, {"formal_eligible": True, "failures": []})
    _write_json(audit_path, {"formal_eligible": True, "blocking_warnings": []})
    return argparse.Namespace(
        dataset="cicids2017",
        train_path=split_dir / "train.parquet",
        val_path=split_dir / "val.parquet",
        test_path=split_dir / "test.parquet",
        audit_path=audit_path,
        raw_dir=raw_dir,
        label_zip=label_zip,
        processed_dir=processed_dir,
        merged_path=processed_dir / "flows_all.parquet",
        attack_coverage_path=attack_coverage_path,
        required_flow_timeout_seconds=120.0,
        required_connection_type_version=1,
        required_tcp_close_policy_version=3,
        required_close_on_tcp_flags=False,
        required_csv_time_offset_hours=3.0,
        required_timestamp_policy_version=2,
        required_window_scope="pcap",
        required_cic_label_max_time_delta_seconds=900.0,
        required_split_parent=split_dir,
        required_view="masked_header_packet",
        required_keep_empty_payload=True,
        allow_missing_audit=False,
    )


def test_formal_gate_accepts_complete_cicids_submode_group(tmp_path: Path) -> None:
    split_dir = (
        tmp_path
        / "data"
        / "processed"
        / "cicids2017"
        / "all_masked_header_split_submode_stratified_group"
    )

    result = gate_module.verify(_args(tmp_path, split_dir))

    assert result["formal_eligible"] is True
    assert result["errors"] == []


def test_formal_gate_rejects_smoke_or_stale_build(tmp_path: Path) -> None:
    split_dir = tmp_path / "data" / "processed" / "cicids2017" / "smoke"
    args = _args(tmp_path, split_dir)
    stale_path = args.processed_dir / "flows_friday_ddos.build.json"
    _write_json(stale_path, {"max_packets_per_flow": 16})

    result = gate_module.verify(args)

    assert result["formal_eligible"] is False
    assert any("smoke" in error for error in result["errors"])
    assert any("missing flow_timeout_seconds" in error for error in result["errors"])
    assert any("timestamp policy" in error for error in result["errors"])


def test_formal_gate_requires_split_audit_unless_explicit_preflight(
    tmp_path: Path,
) -> None:
    split_dir = (
        tmp_path
        / "data"
        / "processed"
        / "cicids2017"
        / "all_masked_header_split_submode_stratified_group"
    )
    args = _args(tmp_path, split_dir)
    args.audit_path = None

    result = gate_module.verify(args)

    assert result["formal_eligible"] is False
    assert any("requires --audit-path" in error for error in result["errors"])

    args.allow_missing_audit = True
    result = gate_module.verify(args)

    assert result["formal_eligible"] is True
