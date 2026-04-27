import pandas as pd

from traffic_bert.data.validate import validate_processed_frame, validation_summary


def test_validate_processed_frame_detects_missing_columns() -> None:
    issues = validate_processed_frame(pd.DataFrame({"flow_id": ["a"]}))

    assert issues
    assert issues[0].code == "missing_columns"
    assert issues[0].severity == "error"


def test_validation_summary_ok_for_minimal_valid_frame() -> None:
    frame = pd.DataFrame(
        [
            {
                "flow_id": "flow-1",
                "source_dataset": "unit",
                "source_file": "unit.pcap",
                "source_label": "DDoS",
                "major_label": "dos_ddos",
                "minor_labels": ["ddos"],
                "split": "train",
                "view": "payload_only",
                "packet_count": 1,
                "payload_byte_length": 3,
                "packet_byte_length": 3,
                "has_payload": True,
                "packet_directions": ["fwd"],
                "packet_lengths": [3],
                "bytes": b"abc",
            }
        ]
    )

    summary = validation_summary(frame)

    assert summary["ok"] is True
    assert summary["issues"] == []

