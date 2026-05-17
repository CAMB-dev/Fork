from pathlib import Path

import pandas as pd
import pytest

from traffic_bert.data.dataset import FlowWindowDataset, flow_collate
from traffic_bert.labels import LabelMap
from traffic_bert.sce import SemanticCodebook, SemanticCodebookEntry


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
                "connection_type": "tcp_payload",
                "context_host_prev_60s_count": 3,
                "context_host_prev_300s_count": 12,
                "context_host_prev_60s_reset_count": 1,
                "context_host_prev_60s_control_count": 2,
                "context_host_prev_60s_payloadless_count": 2,
                "context_host_prev_60s_unique_dst_ports": 2,
                "context_host_prev_60s_unique_dst_hosts": 2,
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

    with_connection = FlowWindowDataset.from_parquet(
        path,
        label_map=label_map,
        view="payload_only",
        max_length=8,
        stride=4,
        use_connection_tokens=True,
    )
    tokens = with_connection.tokenizer.ids_to_tokens(with_connection[0].input_ids[0].tolist())
    assert tokens[1] == "[CONN_TCP_PAYLOAD]"

    with_context = FlowWindowDataset.from_parquet(
        path,
        label_map=label_map,
        view="payload_only",
        max_length=16,
        stride=8,
        use_connection_tokens=True,
        use_context_tokens=True,
    )
    context_tokens = with_context.tokenizer.ids_to_tokens(with_context[0].input_ids[0].tolist())
    assert context_tokens[1] == "[CONN_TCP_PAYLOAD]"
    assert "[CTX_H60_2_4]" in context_tokens
    assert "[CTX_H300_10_19]" in context_tokens

    with_features = FlowWindowDataset.from_parquet(
        path,
        label_map=label_map,
        view="payload_only",
        max_length=8,
        stride=4,
        use_context_features=True,
    )
    feature_batch = flow_collate([with_features[0]])
    assert feature_batch["context_features"].shape == (1, 7)
    assert feature_batch["context_features"][0, 0].item() == pytest.approx(1.386294, rel=1e-5)

    codebook = SemanticCodebook(
        chunk_size=2,
        entries=(SemanticCodebookEntry(token="[SCE_0000]", pattern_hex="6162", count=1),),
    )
    with_sce = FlowWindowDataset.from_parquet(
        path,
        label_map=label_map,
        view="payload_only",
        max_length=8,
        stride=4,
        semantic_codebook=codebook,
    )
    sce_tokens = with_sce.tokenizer.ids_to_tokens(with_sce[0].input_ids[0].tolist())
    assert "[SCE_0000]" in sce_tokens
    assert "b_63" in sce_tokens
