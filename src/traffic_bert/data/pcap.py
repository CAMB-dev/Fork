"""PCAP parsing and flow reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from traffic_bert.data.schema import FlowRecord, PacketViews


def _endpoint(ip: str, port: int) -> str:
    return f"{ip}:{port}"


def _flow_lookup_key(src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: str) -> tuple:
    left = _endpoint(src_ip, src_port)
    right = _endpoint(dst_ip, dst_port)
    a, b = sorted([left, right])
    return proto, a, b


def _flow_id(source_file: str, lookup_key: tuple) -> str:
    digest = hashlib.sha1(f"{source_file}|{lookup_key}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(source_file).stem}_{digest}"


def _payload_and_proto(packet: Any) -> tuple[str, int, int, bytes] | None:
    from scapy.layers.inet import TCP, UDP

    if TCP in packet:
        layer = packet[TCP]
        return "tcp", int(layer.sport), int(layer.dport), bytes(layer.payload)
    if UDP in packet:
        layer = packet[UDP]
        return "udp", int(layer.sport), int(layer.dport), bytes(layer.payload)
    return None


def _mask_packet(packet: Any) -> bytes:
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.l2 import Ether

    masked = packet.copy()
    if Ether in masked:
        masked[Ether].src = "00:00:00:00:00:00"
        masked[Ether].dst = "00:00:00:00:00:00"
    if IP in masked:
        masked[IP].src = "0.0.0.0"
        masked[IP].dst = "0.0.0.0"
        masked[IP].chksum = 0
    if TCP in masked:
        masked[TCP].sport = 0
        masked[TCP].dport = 0
        masked[TCP].chksum = 0
    if UDP in masked:
        masked[UDP].sport = 0
        masked[UDP].dport = 0
        masked[UDP].chksum = 0
    return bytes(masked)


class PcapFlowExtractor:
    """Extract bidirectional TCP/UDP flows from PCAP files."""

    def __init__(self, max_packets_per_flow: int | None = None) -> None:
        self.max_packets_per_flow = max_packets_per_flow

    def extract(self, pcap_path: str | Path) -> list[FlowRecord]:
        from scapy.all import PcapReader
        from scapy.layers.inet import IP

        pcap_path = Path(pcap_path)
        flows: dict[tuple, FlowRecord] = {}
        forward_endpoint: dict[tuple, tuple[str, int, str, int]] = {}

        with PcapReader(str(pcap_path)) as reader:
            for packet in reader:
                if IP not in packet:
                    continue

                transport = _payload_and_proto(packet)
                if transport is None:
                    continue
                proto, src_port, dst_port, payload = transport
                src_ip = str(packet[IP].src)
                dst_ip = str(packet[IP].dst)
                lookup_key = _flow_lookup_key(src_ip, dst_ip, src_port, dst_port, proto)

                if lookup_key not in forward_endpoint:
                    forward_endpoint[lookup_key] = (src_ip, src_port, dst_ip, dst_port)

                fwd = forward_endpoint[lookup_key]
                direction = (
                    "fwd"
                    if (src_ip, src_port, dst_ip, dst_port) == fwd
                    else "bwd"
                )

                timestamp = float(packet.time)
                if lookup_key not in flows:
                    _, endpoint_a, endpoint_b = lookup_key
                    flows[lookup_key] = FlowRecord(
                        flow_id=_flow_id(str(pcap_path), lookup_key),
                        source_file=str(pcap_path),
                        protocol=proto,
                        endpoint_a=endpoint_a,
                        endpoint_b=endpoint_b,
                        start_time=timestamp,
                        end_time=timestamp,
                        packets=[],
                    )

                flow = flows[lookup_key]
                if (
                    self.max_packets_per_flow is not None
                    and len(flow.packets) >= self.max_packets_per_flow
                ):
                    continue

                flow.packets.append(
                    PacketViews(
                        timestamp=timestamp,
                        direction=direction,
                        payload_only=payload,
                        full_packet=bytes(packet),
                        masked_header_packet=_mask_packet(packet),
                    )
                )
                flow.end_time = timestamp

        return sorted(flows.values(), key=lambda item: (item.start_time, item.flow_id))

