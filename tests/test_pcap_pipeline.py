from pathlib import Path

import pandas as pd
import pytest

from traffic_bert.data.build import BuildConfig, build_config_from_yaml, build_processed_dataset
from traffic_bert.data.cic import CicFlowLabelIndex
from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.schema import InputView


def test_pcap_flow_extractor_bidirectional(tmp_path: Path) -> None:
    scapy = pytest.importorskip("scapy.all")

    pcap_path = tmp_path / "sample.pcap"
    packets = [
        scapy.Ether()
        / scapy.IP(src="10.0.0.1", dst="10.0.0.2")
        / scapy.TCP(sport=1234, dport=80)
        / scapy.Raw(b"GET / HTTP/1.1"),
        scapy.Ether()
        / scapy.IP(src="10.0.0.2", dst="10.0.0.1")
        / scapy.TCP(sport=80, dport=1234)
        / scapy.Raw(b"HTTP/1.1 200 OK"),
    ]
    scapy.wrpcap(str(pcap_path), packets)

    flows = PcapFlowExtractor().extract(pcap_path)

    assert len(flows) == 1
    assert flows[0].packet_count == 2
    assert [packet.direction for packet in flows[0].packets] == ["fwd", "bwd"]
    assert flows[0].payload_byte_length > 0
    assert flows[0].packet_byte_length > flows[0].payload_byte_length


def test_build_processed_dataset_from_pcap(tmp_path: Path) -> None:
    scapy = pytest.importorskip("scapy.all")

    pcap_path = tmp_path / "DDoS.pcap"
    packets = [
        scapy.Ether()
        / scapy.IP(src="192.168.1.1", dst="192.168.1.2")
        / scapy.UDP(sport=1111, dport=2222)
        / scapy.Raw(b"payload")
    ]
    scapy.wrpcap(str(pcap_path), packets)

    output_path = tmp_path / "train.parquet"
    stats = build_processed_dataset(
        BuildConfig(
            input_path=pcap_path,
            output_path=output_path,
            label_map_path=Path("configs/label_map.yaml"),
            source_dataset="unit",
            split="train",
            label_source="filename",
            views=(InputView.PAYLOAD_ONLY, InputView.MASKED_HEADER_PACKET),
        )
    )
    frame = pd.read_parquet(output_path)

    assert stats["flows"] == 1
    assert set(frame["view"]) == {"payload_only", "masked_header_packet"}
    assert set(frame["major_label"]) == {"dos_ddos"}


def test_build_processed_dataset_with_cic_csv_labels(tmp_path: Path) -> None:
    scapy = pytest.importorskip("scapy.all")

    pcap_path = tmp_path / "traffic.pcap"
    packets = [
        scapy.Ether()
        / scapy.IP(src="10.0.0.1", dst="10.0.0.2")
        / scapy.TCP(sport=1234, dport=80)
        / scapy.Raw(b"payload")
    ]
    scapy.wrpcap(str(pcap_path), packets)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Label": "DoS Hulk",
            }
        ]
    ).to_csv(labels_path, index=False)

    output_path = tmp_path / "train.parquet"
    build_processed_dataset(
        BuildConfig(
            input_path=pcap_path,
            output_path=output_path,
            label_map_path=Path("configs/label_map.yaml"),
            source_dataset="cic-unit",
            label_source="cic_csv",
            label_csv_path=labels_path,
            views=(InputView.PAYLOAD_ONLY,),
        )
    )

    frame = pd.read_parquet(output_path)
    assert frame["source_label"].tolist() == ["DoS Hulk"]
    assert frame["minor_labels"].iloc[0] == ["dos_hulk"]


def test_build_processed_dataset_counts_unmatched_and_empty_payload_drops(
    tmp_path: Path,
) -> None:
    scapy = pytest.importorskip("scapy.all")

    pcap_path = tmp_path / "traffic.pcap"
    packets = [
        scapy.Ether()
        / scapy.IP(src="10.0.0.1", dst="10.0.0.2")
        / scapy.TCP(sport=1234, dport=80)
        / scapy.Raw(b"payload"),
        scapy.Ether()
        / scapy.IP(src="10.0.0.3", dst="10.0.0.4")
        / scapy.TCP(sport=1235, dport=80),
        scapy.Ether()
        / scapy.IP(src="10.0.0.5", dst="10.0.0.6")
        / scapy.TCP(sport=1236, dport=80)
        / scapy.Raw(b"unmatched"),
    ]
    scapy.wrpcap(str(pcap_path), packets)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Label": "DDoS",
            },
            {
                "Source IP": "10.0.0.3",
                "Destination IP": "10.0.0.4",
                "Source Port": 1235,
                "Destination Port": 80,
                "Protocol": 6,
                "Label": "DoS Hulk",
            },
        ]
    ).to_csv(labels_path, index=False)

    output_path = tmp_path / "train.parquet"
    stats = build_processed_dataset(
        BuildConfig(
            input_path=pcap_path,
            output_path=output_path,
            label_map_path=Path("configs/label_map.yaml"),
            source_dataset="cic-unit",
            label_source="cic_csv",
            label_csv_path=labels_path,
            views=(InputView.PAYLOAD_ONLY,),
            keep_empty_payload=False,
            drop_unmatched_labels=True,
        )
    )

    assert stats["extracted_flows_total"] == 3
    assert stats["matched_flows"] == 2
    assert stats["unmatched_label_flows"] == 1
    assert stats["empty_payload_flows"] == 1
    assert stats["dropped_unmatched_label_flows"] == 1
    assert stats["dropped_empty_payload_flows"] == 1
    assert stats["flows"] == 1


def test_cic_label_index_ignores_empty_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Label": "",
            }
        ]
    )

    index = CicFlowLabelIndex.from_frame(frame)

    assert index.lookup("10.0.0.1", "10.0.0.2", 1234, 80, 6) is None


def test_cic_label_index_uses_nearest_timestamp_for_duplicate_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Timestamp": "2017-07-07 10:00:00",
                "Label": "BENIGN",
            },
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Timestamp": "2017-07-07 10:05:00",
                "Label": "DDoS",
            },
        ]
    )

    index = CicFlowLabelIndex.from_frame(frame)

    assert index.lookup(
        "10.0.0.1",
        "10.0.0.2",
        1234,
        80,
        6,
        timestamp=pd.Timestamp("2017-07-07 10:04:59"),
    ) == "DDoS"


def test_build_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "build.yaml"
    output_path = tmp_path / "out.parquet"
    config_path.write_text(
        f"""
label_map: configs/label_map.yaml
build:
  views: [payload_only]
  label_source: static
  static_label: DDoS
datasets:
  - dataset: unit
    raw_path: {tmp_path / "input.pcap"}
    processed_path: {output_path}
""",
        encoding="utf-8",
    )

    configs = build_config_from_yaml(config_path)

    assert len(configs) == 1
    assert configs[0].source_dataset == "unit"
    assert configs[0].static_label == "DDoS"
    assert configs[0].views == (InputView.PAYLOAD_ONLY,)
