from pathlib import Path
import importlib.util
import sys
import zipfile

import pandas as pd
import pytest

from traffic_bert.data.cicids_audit import (
    CicidsCsvSpec,
    build_cicids_coverage_audit,
    label_distribution,
)
from traffic_bert.labels import LabelMap

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cicids2017_all_parallel.py"
SPEC = importlib.util.spec_from_file_location("prepare_cicids2017_all_parallel", SCRIPT_PATH)
assert SPEC is not None
prepare_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare_module
SPEC.loader.exec_module(prepare_module)


def test_label_distribution_maps_source_major_and_minor_with_empty_labels() -> None:
    label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))
    labels = pd.Series(
        [
            " BENIGN ",
            "Web Attack \x96 Brute Force",
            "Web Attack \x96 XSS",
            "Web Attack \x96 Sql Injection",
            None,
            "",
        ]
    )

    result = label_distribution(labels, label_map)

    assert result["source_labels"]["BENIGN"] == 1
    assert result["major_labels"] == {"benign": 1, "web_attack": 3}
    assert result["minor_labels"] == {
        "web_bruteforce": 1,
        "xss": 1,
        "sql_injection": 1,
    }


def test_formal_time_block_groups_merge_visible_and_nearby_tuple_rows() -> None:
    frame = pd.DataFrame(
        {
            "bytes": [b"abc", b"def", b"abc", b"ghi"],
            "packet_lengths": [[3], [3], [3], [3]],
            "packet_directions": [["fwd"], ["fwd"], ["fwd"], ["fwd"]],
            "protocol": ["tcp", "tcp", "udp", "tcp"],
            "endpoint_a": ["10.0.0.1:1", "10.0.0.2:2", "10.0.0.3:3", "10.0.0.9:9"],
            "endpoint_b": ["10.0.0.2:2", "10.0.0.1:1", "10.0.0.4:4", "10.0.0.8:8"],
            "start_time": [10.0, 100.0, 500.0, 1000.0],
        }
    )

    groups = prepare_module._formal_time_block_groups(frame, nearby_threshold_seconds=120.0)

    assert groups[0] == groups[1]
    assert groups[0] == groups[2]
    assert groups[3] != groups[0]


def test_cicids_coverage_audit_reports_csv_and_pcap_presence(tmp_path: Path) -> None:
    raw_dir = tmp_path / "CICIDS2017"
    (raw_dir / "pcaps").mkdir(parents=True)
    (raw_dir / "labels").mkdir()
    (raw_dir / "pcaps" / "Friday-WorkingHours.pcap").write_bytes(b"pcap")
    (raw_dir / "labels" / "cicids2017_friday_ddos.csv").write_text("placeholder")

    csv_path = tmp_path / "ddos.csv"
    pd.DataFrame(
        [
            {" Source IP": "10.0.0.1", " Label": "BENIGN"},
            {" Source IP": "10.0.0.2", " Label": "DDoS"},
            {" Source IP": None, " Label": None},
        ]
    ).to_csv(csv_path, index=False)
    label_zip = raw_dir / "csvs" / "GeneratedLabelledFlows.zip"
    label_zip.parent.mkdir()
    with zipfile.ZipFile(label_zip, "w") as archive:
        archive.write(csv_path, "TrafficLabelling /Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    audit = build_cicids_coverage_audit(
        raw_dir=raw_dir,
        label_zip=label_zip,
        label_map_path=Path("configs/label_map.yaml"),
        jobs=(
            CicidsCsvSpec(
                "friday_ddos",
                "Friday-WorkingHours.pcap",
                ("Friday", "DDos"),
                ("DDoS",),
            ),
        ),
    )

    item = audit["items"][0]
    assert item["csv_processed"] is True
    assert item["pcap_exists"] is True
    assert item["rows"] == 3
    assert item["valid_label_rows"] == 2
    assert item["empty_label_rows"] == 1
    assert item["major_labels"] == {"benign": 1, "dos_ddos": 1}


def test_attack_coverage_validator_rejects_single_matched_attack_flow(tmp_path: Path) -> None:
    shard_path = tmp_path / "flows_friday_ddos.parquet"
    pd.DataFrame({"source_label": ["DDoS"]}).to_parquet(shard_path, index=False)
    summary = {
        "slug": "friday_ddos",
        "attack_labels": ["DDoS"],
        "label_counts": {"DDoS": 128027},
    }

    with pytest.raises(RuntimeError, match="match ratio"):
        prepare_module._validate_attack_coverage(
            [summary],
            [shard_path],
            min_attack_flows_per_label=100,
            min_attack_match_ratio=0.01,
        )


def test_attack_coverage_validator_reports_low_support_exception(tmp_path: Path) -> None:
    shard_path = tmp_path / "flows_heartbleed.parquet"
    pd.DataFrame({"source_label": ["Heartbleed"]}).to_parquet(shard_path, index=False)
    summary = {
        "slug": "wednesday_dos",
        "attack_labels": ["Heartbleed"],
        "label_counts": {"Heartbleed": 11},
    }

    report = prepare_module._validate_attack_coverage(
        [summary],
        [shard_path],
        min_attack_flows_per_label=100,
        min_attack_match_ratio=0.01,
    )

    assert report["formal_eligible"] is True
    assert report["items"][0]["low_support_exception"] is True


def test_attack_coverage_validator_allows_low_source_support_when_major_is_supported(
    tmp_path: Path,
) -> None:
    shard_path = tmp_path / "flows_thursday_web.parquet"
    pd.DataFrame(
        {
            "source_label": ["Web Attack XSS"] * 26 + ["Web Attack Brute Force"] * 158,
            "major_label": ["web_attack"] * 184,
        }
    ).to_parquet(shard_path, index=False)
    summary = {
        "slug": "thursday_web",
        "attack_labels": ["Web Attack XSS"],
        "label_counts": {"Web Attack XSS": 652},
    }

    report = prepare_module._validate_attack_coverage(
        [summary],
        [shard_path],
        min_attack_flows_per_label=100,
        min_attack_match_ratio=0.01,
    )

    assert report["formal_eligible"] is True
    assert report["items"][0]["major_support_exception"] is True
    assert report["items"][0]["processed_major_flows"] == 184
    assert "low-support diagnostic only" in report["warnings"][0]


def test_full_day_timestamp_parser_repairs_missing_pm_marker() -> None:
    values = pd.Series(
        [
            "7/4/2017 2:09",
            "7/4/2017 7:11",
            "7/4/2017 9:17",
            "7/4/2017 12:30",
        ]
    )

    parsed = prepare_module._parse_cicids_timestamps(
        values,
        "TrafficLabelling /Tuesday-WorkingHours.pcap_ISCX.csv",
    )

    assert parsed.dt.strftime("%H:%M").tolist() == [
        "14:09",
        "19:11",
        "09:17",
        "12:30",
    ]


def test_afternoon_timestamp_parser_keeps_existing_afternoon_policy() -> None:
    values = pd.Series(["7/7/2017 1:05", "7/7/2017 3:56", "7/7/2017 12:15"])

    parsed = prepare_module._parse_cicids_timestamps(
        values,
        "TrafficLabelling /Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    )

    assert parsed.dt.strftime("%H:%M").tolist() == ["13:05", "15:56", "12:15"]


def test_stale_shard_detection_requires_timestamp_policy_metadata(tmp_path: Path) -> None:
    shard_path = tmp_path / "flows_tuesday_patator.parquet"
    pd.DataFrame({"source_label": ["BENIGN"]}).to_parquet(shard_path, index=False)
    shard_path.with_suffix(".build.json").write_text(
        '{"flow_timeout_seconds": 120.0, "max_packets_per_flow": 16}',
        encoding="utf-8",
    )
    payload = {
        "max_packets_per_flow": 16,
        "max_packets_to_read": None,
        "flow_timeout_seconds": 120.0,
        "min_packet_time": 1.0,
        "max_packet_time": 2.0,
        "csv_time_offset_hours": 3.0,
        "cicids_timestamp_policy_version": prepare_module.TIMESTAMP_POLICY_VERSION,
        "cic_label_max_time_delta_seconds": 900.0,
        "window_scope": "pcap",
    }

    assert prepare_module._is_fresh_shard(shard_path, payload) is False
