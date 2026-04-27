from pathlib import Path

import pandas as pd
import pytest

from traffic_bert.data.build import BuildConfig, build_processed_dataset
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

