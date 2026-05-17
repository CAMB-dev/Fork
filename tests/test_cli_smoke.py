from pathlib import Path

import pandas as pd
import torch
from typer.testing import CliRunner

from traffic_bert.cli import (
    _load_host_window_policies,
    _summarize_pcap_predictions,
    _write_classifier_predictions,
    app,
)
from traffic_bert.labels import LabelMap
from traffic_bert.sce import SemanticCodebook, SemanticCodebookEntry
from traffic_bert.tokenizer import ByteTokenizer


def _write_processed_fixture(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "flow_id": "flow-1",
                "source_dataset": "unit",
                "source_file": "one.pcap",
                "source_label": "BENIGN",
                "major_label": "benign",
                "minor_labels": [],
                "split": "train",
                "view": "payload_only",
                "protocol": "tcp",
                "endpoint_a": "1.1.1.1:1",
                "endpoint_b": "2.2.2.2:2",
                "packet_count": 1,
                "payload_byte_length": 3,
                "packet_byte_length": 3,
                "has_payload": True,
                "start_time": 1.0,
                "end_time": 1.0,
                "packet_directions": ["fwd"],
                "packet_lengths": [3],
                "bytes": b"abc",
            },
            {
                "flow_id": "flow-2",
                "source_dataset": "unit",
                "source_file": "two.pcap",
                "source_label": "DDoS",
                "major_label": "dos_ddos",
                "minor_labels": ["ddos"],
                "split": "train",
                "view": "payload_only",
                "protocol": "tcp",
                "endpoint_a": "3.3.3.3:3",
                "endpoint_b": "4.4.4.4:4",
                "packet_count": 1,
                "payload_byte_length": 3,
                "packet_byte_length": 3,
                "has_payload": True,
                "start_time": 1.0,
                "end_time": 1.0,
                "packet_directions": ["fwd"],
                "packet_lengths": [3],
                "bytes": b"def",
            },
        ]
    ).to_parquet(path, index=False)


def test_data_stats_validate_split_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    parquet = tmp_path / "data.parquet"
    out_dir = tmp_path / "out"
    _write_processed_fixture(parquet)

    stats_result = runner.invoke(app, ["data", "stats", "--input-path", str(parquet), "--output-dir", str(out_dir)])
    validate_result = runner.invoke(app, ["data", "validate", "--input-path", str(parquet)])
    split_result = runner.invoke(app, ["data", "split", "--input-path", str(parquet), "--output-dir", str(out_dir / "split")])
    merge_result = runner.invoke(
        app,
        [
            "data",
            "merge",
            "--input-path",
            str(out_dir / "split" / "train.parquet"),
            "--input-path",
            str(out_dir / "split" / "test.parquet"),
            "--output-path",
            str(out_dir / "merged.parquet"),
        ],
    )

    assert stats_result.exit_code == 0
    assert validate_result.exit_code == 0
    assert split_result.exit_code == 0
    assert merge_result.exit_code == 0
    assert (out_dir / "stats.json").exists()
    assert (out_dir / "class_distribution.csv").exists()
    assert (out_dir / "split" / "train.parquet").exists()
    assert (out_dir / "merged.parquet").exists()


def test_pcap_prediction_summary_counts_high_risk_flows() -> None:
    summary = _summarize_pcap_predictions(
        [
            {
                "flow_id": "a",
                "major_label": "benign",
                "major_prob": 0.99,
                "source_file": "sample.pcap",
                "initiator_endpoint": "10.0.0.1:1234",
                "connection_type": "tcp_payload",
                "start_time": 100.0,
            },
            {
                "flow_id": "b",
                "major_label": "botnet_malware",
                "major_prob": 0.93,
                "major_probs": {"botnet_malware": 0.93, "benign": 0.01},
                "source_file": "sample.pcap",
                "initiator_endpoint": "10.0.0.2:1234",
                "connection_type": "tcp_reset_or_refused",
                "start_time": 105.0,
            },
            {
                "flow_id": "c",
                "major_label": "scan",
                "major_prob": 0.72,
                "major_probs": {"botnet_malware": 0.91, "scan": 0.72},
                "source_file": "sample.pcap",
                "initiator_endpoint": "10.0.0.2:2345",
                "connection_type": "tcp_control_only",
                "start_time": 108.0,
            },
        ],
        risk_threshold=0.8,
        top_flows=1,
        host_window_policies=[
            {"target_label": "botnet_malware", "window_seconds": 30, "threshold": 0.9}
        ],
    )

    assert summary["flow_count"] == 3
    assert summary["predicted_benign_flow_count"] == 1
    assert summary["predicted_attack_flow_count"] == 2
    assert summary["high_risk_flow_count"] == 1
    assert summary["top_high_risk_flows"][0]["flow_id"] == "b"
    assert summary["connection_type_counts"]["tcp_reset_or_refused"] == 1
    assert summary["top_host_risks"][0]["host"] == "10.0.0.2"
    assert summary["top_host_risks"][0]["predicted_attack_flow_count"] == 2
    window = summary["top_host_windows"]["30"][0]
    assert window["host"] == "10.0.0.2"
    assert window["window_seconds"] == 30
    assert window["flow_count"] == 2
    assert window["predicted_attack_flow_count"] == 2
    assert window["high_risk_flow_count"] == 1
    policy_window = summary["policy_host_windows"]["botnet_malware@30s"][0]
    assert policy_window["host"] == "10.0.0.2"
    assert policy_window["policy_hit_flow_count"] == 2
    assert policy_window["target_argmax_flow_count"] == 1
    assert policy_window["policy_triggered"] is True


def test_load_host_window_policy_from_analysis(tmp_path: Path) -> None:
    path = tmp_path / "bot_host_window_analysis.json"
    path.write_text(
        """
        {
          "target_label": "botnet_malware",
          "min_recall": 0.8,
          "min_f1": 0.75,
          "bucket_window_detection": {
            "30": {"val_at_chosen": {"threshold": 0.938}},
            "60": {"val_scan": {"chosen": {"threshold": 0.95}}},
            "300": {
              "val_at_chosen": {"threshold": 0.97},
              "test_at_chosen": {"recall": 0.7, "f1": 0.6}
            }
          }
        }
        """,
        encoding="utf-8",
    )

    policies = _load_host_window_policies(path)

    assert policies == [
        {
            "target_label": "botnet_malware",
            "window_seconds": 30,
            "threshold": 0.938,
            "source": str(path),
        },
        {
            "target_label": "botnet_malware",
            "window_seconds": 60,
            "threshold": 0.95,
            "source": str(path),
        },
    ]


def test_load_host_window_policy_filters_disabled_policies(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        """
        {
          "policies": [
            {
              "target_label": "botnet_malware",
              "window_seconds": 30,
              "threshold": 0.938,
              "enabled": true
            },
            {
              "target_label": "botnet_malware",
              "window_seconds": 300,
              "threshold": 0.972,
              "enabled": false
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    policies = _load_host_window_policies(path)

    assert len(policies) == 1
    assert policies[0]["window_seconds"] == 30


def test_write_classifier_predictions_parquet(tmp_path: Path) -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    output_path = tmp_path / "predictions.parquet"

    _write_classifier_predictions(
        output_path,
        flow_ids=["flow-1", "flow-2"],
        y_true=[label_map.major_id("benign"), label_map.major_id("dos_ddos")],
        y_pred=[label_map.major_id("benign"), label_map.major_id("botnet_malware")],
        major_prob=torch.zeros(2, len(label_map.major_labels)),
        label_map=label_map,
    )

    frame = pd.read_parquet(output_path)
    assert list(frame["flow_id"]) == ["flow-1", "flow-2"]
    assert list(frame["true_major_label"]) == ["benign", "dos_ddos"]
    assert list(frame["pred_major_label"]) == ["benign", "botnet_malware"]
    assert "prob_botnet_malware" in frame.columns


def test_train_classifier_accepts_sce_codebook(tmp_path: Path) -> None:
    runner = CliRunner()
    parquet = tmp_path / "data.parquet"
    output_dir = tmp_path / "classifier"
    codebook_path = tmp_path / "sce_codebook.json"
    _write_processed_fixture(parquet)
    SemanticCodebook(
        chunk_size=2,
        entries=(SemanticCodebookEntry(token="[SCE_0000]", pattern_hex="6162", count=1),),
    ).save(codebook_path)

    result = runner.invoke(
        app,
        [
            "train",
            "classifier",
            "--train-path",
            str(parquet),
            "--output-dir",
            str(output_dir),
            "--view",
            "payload_only",
            "--epochs",
            "0",
            "--batch-size",
            "1",
            "--max-length",
            "8",
            "--max-windows",
            "1",
            "--device",
            "cpu",
            "--semantic-codebook-path",
            str(codebook_path),
        ],
    )

    assert result.exit_code == 0
    payload = torch.load(output_dir / "classifier.pt", map_location="cpu")
    assert payload["semantic_codebook"]["entries"][0]["token"] == "[SCE_0000]"
    assert payload["model_config"]["vocab_size"] == ByteTokenizer(extra_tokens=["[SCE_0000]"]).vocab_size
