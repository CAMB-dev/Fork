from pathlib import Path

import pandas as pd

from traffic_bert.data.dataset import FlowWindowDataset, flow_collate
from traffic_bert.labels import LabelMap


def test_flow_window_dataset_and_collate(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
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
            }
        ]
    )
    frame.to_parquet(path, index=False)
    label_map = LabelMap.from_yaml("configs/label_map.yaml")

    dataset = FlowWindowDataset.from_parquet(
        path,
        label_map=label_map,
        view="payload_only",
        max_length=8,
        stride=4,
    )
    batch = flow_collate([dataset[0]])

    assert batch["input_ids"].shape == (1, 1, 8)
    assert batch["major_labels"].item() == label_map.major_id("dos_ddos")
    assert batch["minor_labels"][0, label_map.minor_to_id["ddos"]].item() == 1.0

