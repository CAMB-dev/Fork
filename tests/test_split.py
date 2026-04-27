import pandas as pd

from traffic_bert.data.split import assign_file_time_split, processed_stats


def test_assign_file_time_split_keeps_source_file_together() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c", "d"],
            "source_file": ["one.pcap", "one.pcap", "two.pcap", "two.pcap"],
        }
    )

    result = assign_file_time_split(frame)

    assert result[result["source_file"] == "one.pcap"]["split"].nunique() == 1
    assert result[result["source_file"] == "two.pcap"]["split"].nunique() == 1


def test_processed_stats_counts_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": ["a", "b"],
            "split": ["train", "test"],
            "view": ["payload_only", "payload_only"],
            "major_label": ["benign", "dos_ddos"],
            "source_dataset": ["unit", "unit"],
            "packet_count": [1, 3],
            "payload_byte_length": [10, 20],
            "packet_byte_length": [30, 40],
        }
    )

    stats = processed_stats(frame)

    assert stats["rows"] == 2
    assert stats["flows"] == 2
    assert stats["major_labels"]["benign"] == 1

