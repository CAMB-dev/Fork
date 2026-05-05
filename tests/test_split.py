import pandas as pd

from traffic_bert.data.split import (
    assign_file_time_split,
    assign_stratified_hash_split,
    assign_time_block_split,
    assign_time_ordered_split,
    processed_stats,
    split_run_stats,
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
    assert stats["split_support"]["major_labels"]["train"]["benign"] == 1


def test_split_run_stats_records_sampling_and_method_metadata() -> None:
    input_frame = pd.DataFrame(
        {
            "flow_id": ["a", "b", "c"],
            "split": ["train", "val", "test"],
            "major_label": ["benign", "benign", "dos_ddos"],
            "source_label": ["BENIGN", "BENIGN", "DDoS"],
            "minor_labels": [[], [], ["ddos"]],
        }
    )
    output_frame = input_frame.iloc[:2].copy()

    stats = split_run_stats(
        input_frame=input_frame,
        output_frame=output_frame,
        split_method="time_block",
        max_per_class=50_000,
        block_size=128,
    )

    assert stats["split_method"] == "time_block"
    assert stats["max_per_class"] == 50_000
    assert stats["block_size"] == 128
    assert stats["sampling"]["input_rows"] == 3
    assert stats["sampling"]["output_rows"] == 2


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


def test_assign_time_ordered_split_orders_within_each_label() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": [f"{label}-{idx}" for label in ["a", "b"] for idx in range(10)],
            "source_label": [label for label in ["a", "b"] for _ in range(10)],
            "start_time": list(range(10)) + list(range(100, 110)),
        }
    )

    result = assign_time_ordered_split(
        frame,
        group_column="flow_id",
        stratify_column="source_label",
        time_column="start_time",
    )

    assert result.groupby("flow_id")["split"].nunique().max() == 1
    for _, group in result.groupby("source_label"):
        train_max = group[group["split"] == "train"]["start_time"].max()
        val_min = group[group["split"] == "val"]["start_time"].min()
        val_max = group[group["split"] == "val"]["start_time"].max()
        test_min = group[group["split"] == "test"]["start_time"].min()
        assert train_max < val_min
        assert val_max < test_min


def test_assign_time_block_split_keeps_nearby_blocks_together() -> None:
    frame = pd.DataFrame(
        {
            "flow_id": [f"{label}-{idx}" for label in ["a", "b"] for idx in range(12)],
            "source_label": [label for label in ["a", "b"] for _ in range(12)],
            "start_time": list(range(12)) + list(range(100, 112)),
        }
    )

    result = assign_time_block_split(
        frame,
        group_column="flow_id",
        stratify_column="source_label",
        time_column="start_time",
        block_size=3,
        seed=7,
    )

    assert result.groupby("flow_id")["split"].nunique().max() == 1
    for _, group in result.groupby("source_label"):
        counts = group["split"].value_counts()
        assert {"train", "val", "test"} <= set(counts.index)
        group = group.sort_values("start_time")
        block_ids = list(range(len(group)))
        group = group.assign(block=[idx // 3 for idx in block_ids])
        assert group.groupby("block")["split"].nunique().max() == 1


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
