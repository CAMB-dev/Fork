import pandas as pd

from traffic_bert.data.split import (
    assign_file_time_split,
    assign_stratified_hash_split,
    processed_stats,
    stratified_sample,
)


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
            "minor_labels": [[], ["ddos"]],
            "source_label": ["BENIGN", "DDoS"],
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
    assert stats["minor_labels"]["ddos"] == 1
    assert stats["source_labels"]["BENIGN"] == 1


def test_assign_stratified_hash_split_keeps_groups_and_labels_present() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": [f"{label}-{idx}" for label in ["a", "b"] for idx in range(20)],
            "source_label": [label for label in ["a", "b"] for _ in range(20)],
        }
    )

    result = assign_stratified_hash_split(
        frame,
        group_column="flow_id",
        stratify_column="source_label",
    )

    assert result.groupby("flow_id")["split"].nunique().max() == 1
    counts = result.groupby(["source_label", "split"]).size().unstack(fill_value=0)
    assert set(counts.columns) == {"train", "val", "test"}
    assert (counts > 0).all().all()


def test_stratified_sample_caps_each_class() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": [f"f{i}" for i in range(8)],
            "major_label": ["benign"] * 5 + ["dos_ddos"] * 3,
            "start_time": list(range(8)),
        }
    )

    sampled = stratified_sample(frame, max_per_class=2, seed=1)

    assert sampled["major_label"].value_counts().to_dict() == {
        "benign": 2,
        "dos_ddos": 2,
    }
