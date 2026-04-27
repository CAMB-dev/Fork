from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from traffic_bert.cli import app


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

    assert stats_result.exit_code == 0
    assert validate_result.exit_code == 0
    assert split_result.exit_code == 0
    assert (out_dir / "stats.json").exists()
    assert (out_dir / "class_distribution.csv").exists()
    assert (out_dir / "split" / "train.parquet").exists()

