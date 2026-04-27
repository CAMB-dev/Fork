from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from traffic_bert.cli import app
from traffic_bert.data.payload_csv import PayloadCsvBuildConfig, build_payload_csv_dataset


def _write_payload_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "payload_byte_1": 1,
                "payload_byte_2": 2,
                "payload_byte_3": 0,
                "payload_byte_4": 0,
                "ttl": 64,
                "total_len": 4,
                "protocol": "tcp",
                "t_delta": 0.1,
                "label": "BENIGN",
            },
            {
                "payload_byte_1": 9,
                "payload_byte_2": 0,
                "payload_byte_3": 8,
                "payload_byte_4": 0,
                "ttl": 128,
                "total_len": 4,
                "protocol": "udp",
                "t_delta": 0.2,
                "label": "reconnaissance",
            },
            {
                "payload_byte_1": 0,
                "payload_byte_2": 0,
                "payload_byte_3": 0,
                "payload_byte_4": 0,
                "ttl": 1,
                "total_len": 0,
                "protocol": "others",
                "t_delta": 0.0,
                "label": "normal",
            },
        ]
    ).to_csv(path, index=False)


def test_build_payload_csv_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "payload.csv"
    output_path = tmp_path / "payload.parquet"
    _write_payload_csv(input_path)

    stats = build_payload_csv_dataset(
        PayloadCsvBuildConfig(
            input_path=input_path,
            output_path=output_path,
            label_map_path=Path("configs/label_map.yaml"),
            source_dataset="payload-unit",
            chunksize=2,
        )
    )
    frame = pd.read_parquet(output_path)

    assert stats["rows_read"] == 3
    assert stats["rows_written"] == 2
    assert stats["dropped_empty_payload"] == 1
    assert frame["bytes"].tolist() == [b"\x01\x02", b"\x09\x00\x08"]
    assert frame["major_label"].tolist() == ["benign", "scan"]
    assert [list(value) for value in frame["minor_labels"]] == [[], ["reconnaissance"]]
    assert [list(value) for value in frame["packet_lengths"]] == [[2], [3]]


def test_build_payload_csv_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    input_path = tmp_path / "payload.csv"
    output_path = tmp_path / "payload.parquet"
    _write_payload_csv(input_path)

    result = runner.invoke(
        app,
        [
            "data",
            "build-payload-csv",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--source-dataset",
            "payload-unit",
            "--chunksize",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
