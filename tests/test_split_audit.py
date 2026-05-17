from pathlib import Path
import importlib.util
import subprocess
import sys

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_processed_split.py"
SPEC = importlib.util.spec_from_file_location("audit_processed_split", SCRIPT_PATH)
assert SPEC is not None
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_module)
audit = audit_module.audit


def _write_split(path: Path, rows: int) -> None:
    pd.DataFrame(
        {
            "flow_id": [f"{path.stem}-{idx}" for idx in range(rows)],
            "source_file": ["friday.pcap"] * rows,
            "source_label": ["BENIGN"] * rows,
            "major_label": ["benign"] * rows,
            "minor_labels": [[] for _ in range(rows)],
            "view": ["payload_only"] * rows,
            "protocol": ["tcp"] * rows,
            "endpoint_a": [f"10.0.0.{idx}:1234" for idx in range(rows)],
            "endpoint_b": ["10.0.0.100:80"] * rows,
            "start_time": [float(idx * 120) for idx in range(rows)],
            "packet_count": [2] * rows,
            "payload_byte_length": [128] * rows,
        }
    ).to_parquet(path, index=False)


def test_audit_warns_when_split_contains_only_one_major_label(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, 3)
    _write_split(val_path, 2)
    _write_split(test_path, 2)

    result = audit(train_path, val_path, test_path)

    assert result["all_major_labels"] == ["benign"]
    assert "fewer than 2 major labels across all splits: benign" in result["warnings"]


def test_audit_detects_flow_id_and_nearby_five_tuple_leakage(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    base = {
        "source_file": "friday.pcap",
        "source_label": "DDoS",
        "major_label": "dos_ddos",
        "minor_labels": ["ddos"],
        "view": "payload_only",
        "protocol": "tcp",
        "endpoint_a": "10.0.0.1:1234",
        "endpoint_b": "10.0.0.2:80",
        "packet_count": 2,
        "payload_byte_length": 128,
    }
    pd.DataFrame(
        [
            {**base, "flow_id": "same-flow", "start_time": 100.0},
            {**base, "flow_id": "near-train", "start_time": 200.0},
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame([{**base, "flow_id": "near-val", "start_time": 230.0}]).to_parquet(
        val_path,
        index=False,
    )
    pd.DataFrame([{**base, "flow_id": "same-flow", "start_time": 400.0}]).to_parquet(
        test_path,
        index=False,
    )

    result = audit(train_path, val_path, test_path, nearby_threshold_seconds=60)

    assert result["flow_id_leakage"]["count"] == 1
    assert result["nearby_five_tuple_split_leakage"]["count"] >= 1
    assert any("flow_id leakage" in warning for warning in result["warnings"])


def test_audit_detects_model_visible_byte_overlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    def row(split: str, idx: int, payload: bytes) -> dict:
        return {
            "flow_id": f"{split}-{idx}",
            "source_file": f"{split}.pcap",
            "source_label": "DDoS",
            "major_label": "dos_ddos",
            "minor_labels": ["ddos"],
            "view": "payload_only",
            "protocol": "tcp",
            "endpoint_a": f"10.0.0.{idx}:1234",
            "endpoint_b": "10.0.0.100:80",
            "start_time": float(idx * 120),
            "packet_count": 1,
            "payload_byte_length": len(payload),
            "bytes": payload,
            "packet_lengths": [len(payload)],
            "packet_directions": ["fwd"],
        }

    pd.DataFrame([row("train", 1, b"same-visible-bytes")]).to_parquet(train_path, index=False)
    pd.DataFrame([row("val", 2, b"same-visible-bytes")]).to_parquet(val_path, index=False)
    pd.DataFrame([row("test", 3, b"different")]).to_parquet(test_path, index=False)

    result = audit(train_path, val_path, test_path)

    assert result["bytes_hash_overlap"]["train_vs_val"]["count"] == 1
    assert result["first_window_hash_overlap"]["train_vs_val"]["count"] == 1
    assert result["formal_eligible"] is False
    assert any("first-window token overlap" in warning for warning in result["warnings"])


def test_audit_blocks_train_unseen_ambiguous_attack_submode(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    def rows(split: str, label: str, source_label: str, count: int, *, small_shape: bool) -> list[dict]:
        packet_lengths = [66, 60, 66, 60, 62, 60] if small_shape else [180, 220, 180]
        directions = ["fwd", "bwd", "fwd", "bwd", "fwd", "bwd"] if small_shape else ["fwd", "bwd", "fwd"]
        payload_length = 18 if small_shape else 420
        output = []
        split_offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
        for idx in range(count):
            seed = f"{split}-{label}-{source_label}-{idx}-".encode()
            payload = (seed * ((payload_length // len(seed)) + 1))[:payload_length]
            output.append(
                {
                    "flow_id": f"{split}-{label}-{idx}",
                    "source_file": f"{split}.pcap",
                    "source_label": source_label,
                    "major_label": label,
                    "minor_labels": [] if label == "benign" else [source_label],
                    "view": "masked_header_packet",
                    "protocol": "tcp",
                    "connection_type": "tcp_payload",
                    "endpoint_a": f"10.{split_offset // 100000}.{idx % 250}.1:1234",
                    "endpoint_b": "10.0.0.100:80",
                    "start_time": float(split_offset + idx * 300),
                    "packet_count": len(packet_lengths),
                    "payload_byte_length": payload_length,
                    "bytes": payload,
                    "packet_lengths": packet_lengths,
                    "packet_directions": directions,
                }
            )
        return output

    pd.DataFrame(
        [
            *rows("train", "benign", "BENIGN", 60, small_shape=True),
            *rows("train", "botnet_malware", "Bot", 60, small_shape=False),
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        [
            *rows("val", "benign", "BENIGN", 60, small_shape=False),
            *rows("val", "botnet_malware", "Bot", 60, small_shape=True),
        ]
    ).to_parquet(
        val_path, index=False
    )
    pd.DataFrame(
        [
            *rows("test", "benign", "BENIGN", 60, small_shape=False),
            *rows("test", "botnet_malware", "Bot", 60, small_shape=False),
        ]
    ).to_parquet(
        test_path, index=False
    )

    result = audit(
        train_path,
        val_path,
        test_path,
        low_support_labels=(),
        submode_min_train_support=50,
        submode_min_eval_support=50,
        submode_min_ambiguous_unseen_rows=10,
    )

    assert result["formal_eligible"] is False
    coverage = result["label_submode_train_coverage"]
    assert coverage["blocking_items"]
    assert any("label submode train coverage is weak" in warning for warning in result["warnings"])
    assert any(item["label"] == "botnet_malware" for item in coverage["blocking_items"])


def test_audit_allows_missing_total_one_major_label(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    def row(split: str, idx: int, label: str, payload: bytes) -> dict:
        return {
            "flow_id": f"{split}-{idx}",
            "source_file": f"{split}.pcap",
            "source_label": label,
            "major_label": label,
            "minor_labels": [label] if label != "benign" else [],
            "view": "payload_only",
            "protocol": "tcp",
            "endpoint_a": f"10.0.0.{idx}:1234",
            "endpoint_b": "10.0.0.100:80",
            "start_time": float(idx * 120),
            "packet_count": 1,
            "payload_byte_length": len(payload),
            "bytes": payload,
            "packet_lengths": [len(payload)],
            "packet_directions": ["fwd"],
        }

    pd.DataFrame(
        [
            row("train", 1, "heartbleed", b"heartbleed-only"),
            row("train", 2, "benign", b"benign-train"),
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame([row("val", 3, "benign", b"benign-val")]).to_parquet(
        val_path,
        index=False,
    )
    pd.DataFrame([row("test", 4, "benign", b"benign-test")]).to_parquet(
        test_path,
        index=False,
    )

    result = audit(
        train_path,
        val_path,
        test_path,
        low_support_labels=(),
        ignore_source_file_overlap_for_eligibility=True,
    )

    assert result["formal_eligible"] is True
    assert any("low-support major label heartbleed" in item for item in result["warnings"])
    assert result["blocking_warnings"] == []


def test_audit_cli_fail_on_warnings_exits_nonzero(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, 3)
    _write_split(val_path, 2)
    _write_split(test_path, 2)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--train-path",
            str(train_path),
            "--val-path",
            str(val_path),
            "--test-path",
            str(test_path),
            "--fail-on-warnings",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1


def test_audit_low_support_labels_can_be_dataset_specific(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    def rows(split: str, labels: list[str]) -> list[dict]:
        return [
            {
                "flow_id": f"{split}-{idx}",
                "source_file": f"{split}-{idx}.pcap",
                "source_label": label,
                "major_label": label,
                "minor_labels": [],
                "view": "payload_only",
                "protocol": "tcp",
                "endpoint_a": f"10.0.{idx}.1:1234",
                "endpoint_b": "10.0.0.100:80",
                "start_time": float(idx * 120),
                "packet_count": 1,
                "payload_byte_length": 8,
            }
            for idx, label in enumerate(labels)
        ]

    pd.DataFrame(rows("train", ["benign", "botnet_malware"])).to_parquet(
        train_path,
        index=False,
    )
    pd.DataFrame(rows("val", ["benign", "botnet_malware"])).to_parquet(
        val_path,
        index=False,
    )
    pd.DataFrame(rows("test", ["benign", "botnet_malware"])).to_parquet(
        test_path,
        index=False,
    )

    result = audit(
        train_path,
        val_path,
        test_path,
        low_support_labels=("botnet_malware",),
        low_support_min=1,
    )

    assert result["low_support"]["major_labels"] == {
        "botnet_malware": {"train": 1, "val": 1, "test": 1}
    }
    assert not any("infiltration" in warning for warning in result["warnings"])
