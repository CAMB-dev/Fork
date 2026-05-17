"""PCAP parsing and flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import socket
import struct
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


def _flow_id(source_file: str, lookup_key: tuple, session_index: int, start_time: float) -> str:
    digest = hashlib.sha1(
        f"{source_file}|{lookup_key}|{session_index}|{start_time:.6f}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{Path(source_file).stem}_{digest}_{session_index:06d}"


def _payload_and_proto(packet: Any) -> tuple[str, int, int, bytes] | None:
    from scapy.layers.inet import TCP, UDP

    if TCP in packet:
        layer = packet[TCP]
        return "tcp", int(layer.sport), int(layer.dport), bytes(layer.payload)
    if UDP in packet:
        layer = packet[UDP]
        return "udp", int(layer.sport), int(layer.dport), bytes(layer.payload)
    return None


def _tcp_close_flags(packet: Any) -> bool:
    from scapy.layers.inet import TCP

    if TCP not in packet:
        return False
    flags = str(packet[TCP].flags)
    return "R" in flags or "F" in flags


def _format_tcp_flags(value: int) -> str:
    return "".join(
        flag
        for bit, flag in [
            (0x01, "F"),
            (0x02, "S"),
            (0x04, "R"),
            (0x08, "P"),
            (0x10, "A"),
            (0x20, "U"),
            (0x40, "E"),
            (0x80, "C"),
        ]
        if value & bit
    )


@dataclass(frozen=True)
class _ParsedRawPacket:
    protocol: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    payload: bytes
    full_packet: bytes
    masked_header_packet: bytes
    tcp_flags: str | None = None


def _masked_ipv4_packet(
    raw_packet: bytes,
    *,
    ip_offset: int,
    transport_offset: int,
    protocol: int,
) -> bytes:
    masked = bytearray(raw_packet)
    if len(masked) >= 12:
        masked[0:6] = b"\x00" * 6
        masked[6:12] = b"\x00" * 6
    if len(masked) >= ip_offset + 20:
        masked[ip_offset + 10 : ip_offset + 12] = b"\x00" * 2
        masked[ip_offset + 12 : ip_offset + 20] = b"\x00" * 8
    if protocol == 6 and len(masked) >= transport_offset + 18:
        masked[transport_offset : transport_offset + 4] = b"\x00" * 4
        masked[transport_offset + 16 : transport_offset + 18] = b"\x00" * 2
    elif protocol == 17 and len(masked) >= transport_offset + 8:
        masked[transport_offset : transport_offset + 4] = b"\x00" * 4
        masked[transport_offset + 6 : transport_offset + 8] = b"\x00" * 2
    return bytes(masked)


def _parse_raw_ethernet_ipv4(raw_packet: bytes) -> _ParsedRawPacket | None:
    if len(raw_packet) < 14:
        return None
    ether_type_offset = 12
    ether_type = struct.unpack("!H", raw_packet[ether_type_offset : ether_type_offset + 2])[0]
    ip_offset = 14
    while ether_type in {0x8100, 0x88A8}:
        if len(raw_packet) < ip_offset + 4:
            return None
        ether_type = struct.unpack("!H", raw_packet[ip_offset + 2 : ip_offset + 4])[0]
        ip_offset += 4
    if ether_type != 0x0800 or len(raw_packet) < ip_offset + 20:
        return None

    version_ihl = raw_packet[ip_offset]
    if version_ihl >> 4 != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(raw_packet) < ip_offset + ihl:
        return None
    total_length = struct.unpack("!H", raw_packet[ip_offset + 2 : ip_offset + 4])[0]
    ip_end = min(len(raw_packet), ip_offset + total_length)
    protocol = raw_packet[ip_offset + 9]
    if protocol not in {6, 17}:
        return None
    fragment = struct.unpack("!H", raw_packet[ip_offset + 6 : ip_offset + 8])[0]
    if fragment & 0x1FFF:
        return None

    src_ip = socket.inet_ntoa(raw_packet[ip_offset + 12 : ip_offset + 16])
    dst_ip = socket.inet_ntoa(raw_packet[ip_offset + 16 : ip_offset + 20])
    transport_offset = ip_offset + ihl
    if protocol == 6:
        if ip_end < transport_offset + 20:
            return None
        src_port, dst_port = struct.unpack(
            "!HH", raw_packet[transport_offset : transport_offset + 4]
        )
        tcp_header_length = (raw_packet[transport_offset + 12] >> 4) * 4
        if tcp_header_length < 20 or ip_end < transport_offset + tcp_header_length:
            return None
        payload_offset = transport_offset + tcp_header_length
        tcp_flags = _format_tcp_flags(raw_packet[transport_offset + 13])
        proto_name = "tcp"
    else:
        if ip_end < transport_offset + 8:
            return None
        src_port, dst_port = struct.unpack(
            "!HH", raw_packet[transport_offset : transport_offset + 4]
        )
        payload_offset = transport_offset + 8
        tcp_flags = None
        proto_name = "udp"

    return _ParsedRawPacket(
        protocol=proto_name,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=int(src_port),
        dst_port=int(dst_port),
        payload=raw_packet[payload_offset:ip_end],
        full_packet=raw_packet,
        masked_header_packet=_masked_ipv4_packet(
            raw_packet,
            ip_offset=ip_offset,
            transport_offset=transport_offset,
            protocol=protocol,
        ),
        tcp_flags=tcp_flags,
    )


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

    def __init__(
        self,
        max_packets_per_flow: int | None = None,
        max_packets_to_read: int | None = None,
        max_packets_to_skip: int = 0,
        min_packet_time: float | None = None,
        max_packet_time: float | None = None,
        flow_timeout_seconds: float | None = 120.0,
        close_on_tcp_flags: bool = False,
    ) -> None:
        self.max_packets_per_flow = max_packets_per_flow
        self.max_packets_to_read = max_packets_to_read
        self.max_packets_to_skip = max_packets_to_skip
        self.min_packet_time = min_packet_time
        self.max_packet_time = max_packet_time
        self.flow_timeout_seconds = flow_timeout_seconds
        self.close_on_tcp_flags = close_on_tcp_flags

    def extract(self, pcap_path: str | Path) -> list[FlowRecord]:
        from scapy.all import RawPcapReader

        pcap_path = Path(pcap_path)
        flows: dict[tuple, FlowRecord] = {}
        active_flow_key: dict[tuple, tuple] = {}
        forward_endpoint: dict[tuple, tuple[str, int, str, int]] = {}
        session_counts: dict[tuple, int] = {}
        tcp_fin_directions: dict[tuple, set[str]] = {}

        if self.min_packet_time is not None or self.max_packet_time is not None:
            return self._extract_time_window(
                pcap_path,
                flows,
                active_flow_key,
                forward_endpoint,
                session_counts,
                tcp_fin_directions,
            )

        try:
            reader = RawPcapReader(str(pcap_path))
        except Exception:
            return self._extract_scapy(
                pcap_path,
                flows,
                active_flow_key,
                forward_endpoint,
                session_counts,
                tcp_fin_directions,
            )

        with reader:
            for packet_index, (raw_packet, metadata) in enumerate(reader):
                if packet_index < self.max_packets_to_skip:
                    continue
                if (
                    self.max_packets_to_read is not None
                    and packet_index >= self.max_packets_to_skip + self.max_packets_to_read
                ):
                    break
                self._add_raw_packet(
                    bytes(raw_packet),
                    _raw_metadata_timestamp(metadata),
                    pcap_path,
                    flows,
                    active_flow_key,
                    forward_endpoint,
                    session_counts,
                    tcp_fin_directions,
                )

        return sorted(flows.values(), key=lambda item: (item.start_time, item.flow_id))

    def _extract_scapy(
        self,
        pcap_path: Path,
        flows: dict[tuple, FlowRecord],
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        session_counts: dict[tuple, int],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> list[FlowRecord]:
        from scapy.all import PcapReader

        with PcapReader(str(pcap_path)) as reader:
            for packet_index, packet in enumerate(reader):
                if packet_index < self.max_packets_to_skip:
                    continue
                if (
                    self.max_packets_to_read is not None
                    and packet_index >= self.max_packets_to_skip + self.max_packets_to_read
                ):
                    break
                self._add_packet(
                    packet,
                    pcap_path,
                    flows,
                    active_flow_key,
                    forward_endpoint,
                    session_counts,
                    tcp_fin_directions,
                )

        return sorted(flows.values(), key=lambda item: (item.start_time, item.flow_id))

    def _extract_time_window(
        self,
        pcap_path: Path,
        flows: dict[tuple, FlowRecord],
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        session_counts: dict[tuple, int],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> list[FlowRecord]:
        from scapy.all import RawPcapReader

        decoded_packets = 0
        try:
            reader = RawPcapReader(str(pcap_path))
        except Exception:
            return self._extract_time_window_scapy(
                pcap_path,
                flows,
                active_flow_key,
                forward_endpoint,
                session_counts,
                tcp_fin_directions,
            )

        with reader:
            for _, (raw_packet, metadata) in enumerate(reader):
                timestamp = _raw_metadata_timestamp(metadata)
                if self.min_packet_time is not None and timestamp < self.min_packet_time:
                    continue
                if self.max_packet_time is not None and timestamp > self.max_packet_time:
                    break
                if (
                    self.max_packets_to_read is not None
                    and decoded_packets >= self.max_packets_to_read
                ):
                    break
                decoded_packets += 1
                self._add_raw_packet(
                    bytes(raw_packet),
                    timestamp,
                    pcap_path,
                    flows,
                    active_flow_key,
                    forward_endpoint,
                    session_counts,
                    tcp_fin_directions,
                )

        return sorted(flows.values(), key=lambda item: (item.start_time, item.flow_id))

    def _extract_time_window_scapy(
        self,
        pcap_path: Path,
        flows: dict[tuple, FlowRecord],
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        session_counts: dict[tuple, int],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> list[FlowRecord]:
        from scapy.all import PcapReader

        decoded_packets = 0
        with PcapReader(str(pcap_path)) as reader:
            for packet in reader:
                timestamp = float(packet.time)
                if self.min_packet_time is not None and timestamp < self.min_packet_time:
                    continue
                if self.max_packet_time is not None and timestamp > self.max_packet_time:
                    break
                if (
                    self.max_packets_to_read is not None
                    and decoded_packets >= self.max_packets_to_read
                ):
                    break
                decoded_packets += 1
                self._add_packet(
                    packet,
                    pcap_path,
                    flows,
                    active_flow_key,
                    forward_endpoint,
                    session_counts,
                    tcp_fin_directions,
                )

        return sorted(flows.values(), key=lambda item: (item.start_time, item.flow_id))

    def _add_raw_packet(
        self,
        raw_packet: bytes,
        timestamp: float,
        pcap_path: Path,
        flows: dict[tuple, FlowRecord],
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        session_counts: dict[tuple, int],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> None:
        parsed = _parse_raw_ethernet_ipv4(raw_packet)
        if parsed is None:
            return
        lookup_key = _flow_lookup_key(
            parsed.src_ip,
            parsed.dst_ip,
            parsed.src_port,
            parsed.dst_port,
            parsed.protocol,
        )

        flow_key = active_flow_key.get(lookup_key)
        if flow_key is not None and self.flow_timeout_seconds is not None:
            previous = flows[flow_key]
            if timestamp - previous.end_time > self.flow_timeout_seconds:
                self._drop_active_flow(
                    lookup_key,
                    flow_key,
                    active_flow_key,
                    forward_endpoint,
                    tcp_fin_directions,
                )
                flow_key = None
        if (
            flow_key is not None
            and self._starts_new_tcp_after_half_close(
                flow_key,
                parsed.tcp_flags,
                tcp_fin_directions,
            )
        ):
            self._drop_active_flow(
                lookup_key,
                flow_key,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )
            flow_key = None

        if flow_key is None:
            forward_endpoint[lookup_key] = (
                parsed.src_ip,
                parsed.src_port,
                parsed.dst_ip,
                parsed.dst_port,
            )
            session_index = session_counts.get(lookup_key, 0)
            session_counts[lookup_key] = session_index + 1
            flow_key = (*lookup_key, session_index)
            active_flow_key[lookup_key] = flow_key
            _, endpoint_a, endpoint_b = lookup_key
            fwd = forward_endpoint[lookup_key]
            flows[flow_key] = FlowRecord(
                flow_id=_flow_id(str(pcap_path), lookup_key, session_index, timestamp),
                source_file=str(pcap_path),
                protocol=parsed.protocol,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                initiator_endpoint=_endpoint(fwd[0], fwd[1]),
                responder_endpoint=_endpoint(fwd[2], fwd[3]),
                start_time=timestamp,
                end_time=timestamp,
                packets=[],
            )

        fwd = forward_endpoint[lookup_key]
        direction = (
            "fwd"
            if (
                parsed.src_ip,
                parsed.src_port,
                parsed.dst_ip,
                parsed.dst_port,
            )
            == fwd
            else "bwd"
        )
        flow = flows[flow_key]
        flow.observed_packet_count += 1
        if self.max_packets_per_flow is not None and len(flow.packets) >= self.max_packets_per_flow:
            flow.end_time = timestamp
            self._update_tcp_close_state(
                lookup_key,
                flow_key,
                direction,
                parsed.tcp_flags,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )
            return

        flow.packets.append(
            PacketViews(
                timestamp=timestamp,
                direction=direction,
                payload_only=parsed.payload,
                full_packet=parsed.full_packet,
                masked_header_packet=parsed.masked_header_packet,
                tcp_flags=parsed.tcp_flags,
            )
        )
        flow.end_time = timestamp
        self._update_tcp_close_state(
            lookup_key,
            flow_key,
            direction,
            parsed.tcp_flags,
            active_flow_key,
            forward_endpoint,
            tcp_fin_directions,
        )

    def _drop_active_flow(
        self,
        lookup_key: tuple,
        flow_key: tuple,
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> None:
        active_flow_key.pop(lookup_key, None)
        forward_endpoint.pop(lookup_key, None)
        tcp_fin_directions.pop(flow_key, None)

    def _starts_new_tcp_after_half_close(
        self,
        flow_key: tuple,
        tcp_flags: str | None,
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> bool:
        return (
            self.close_on_tcp_flags
            and bool(tcp_flags)
            and flow_key in tcp_fin_directions
            and "S" in tcp_flags
            and "A" not in tcp_flags
        )

    def _update_tcp_close_state(
        self,
        lookup_key: tuple,
        flow_key: tuple,
        direction: str,
        tcp_flags: str | None,
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> None:
        if not self.close_on_tcp_flags or not tcp_flags:
            return
        if "R" in tcp_flags:
            self._drop_active_flow(
                lookup_key,
                flow_key,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )
            return
        if "F" not in tcp_flags:
            return
        directions = tcp_fin_directions.setdefault(flow_key, set())
        directions.add(direction)
        if len(directions) >= 2:
            self._drop_active_flow(
                lookup_key,
                flow_key,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )

    def _add_packet(
        self,
        packet: Any,
        pcap_path: Path,
        flows: dict[tuple, FlowRecord],
        active_flow_key: dict[tuple, tuple],
        forward_endpoint: dict[tuple, tuple[str, int, str, int]],
        session_counts: dict[tuple, int],
        tcp_fin_directions: dict[tuple, set[str]],
    ) -> None:
        from scapy.layers.inet import IP, TCP

        if IP not in packet:
            return

        transport = _payload_and_proto(packet)
        if transport is None:
            return
        proto, src_port, dst_port, payload = transport
        src_ip = str(packet[IP].src)
        dst_ip = str(packet[IP].dst)
        lookup_key = _flow_lookup_key(src_ip, dst_ip, src_port, dst_port, proto)

        timestamp = float(packet.time)
        tcp_flags = str(packet[TCP].flags) if TCP in packet else None
        flow_key = active_flow_key.get(lookup_key)
        if flow_key is not None and self.flow_timeout_seconds is not None:
            previous = flows[flow_key]
            if timestamp - previous.end_time > self.flow_timeout_seconds:
                self._drop_active_flow(
                    lookup_key,
                    flow_key,
                    active_flow_key,
                    forward_endpoint,
                    tcp_fin_directions,
                )
                flow_key = None
        if (
            flow_key is not None
            and self._starts_new_tcp_after_half_close(
                flow_key,
                tcp_flags,
                tcp_fin_directions,
            )
        ):
            self._drop_active_flow(
                lookup_key,
                flow_key,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )
            flow_key = None

        if flow_key is None:
            forward_endpoint[lookup_key] = (src_ip, src_port, dst_ip, dst_port)
            session_index = session_counts.get(lookup_key, 0)
            session_counts[lookup_key] = session_index + 1
            flow_key = (*lookup_key, session_index)
            active_flow_key[lookup_key] = flow_key
            _, endpoint_a, endpoint_b = lookup_key
            fwd = forward_endpoint[lookup_key]
            flows[flow_key] = FlowRecord(
                flow_id=_flow_id(str(pcap_path), lookup_key, session_index, timestamp),
                source_file=str(pcap_path),
                protocol=proto,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                initiator_endpoint=_endpoint(fwd[0], fwd[1]),
                responder_endpoint=_endpoint(fwd[2], fwd[3]),
                start_time=timestamp,
                end_time=timestamp,
                packets=[],
            )

        fwd = forward_endpoint[lookup_key]
        direction = "fwd" if (src_ip, src_port, dst_ip, dst_port) == fwd else "bwd"
        flow = flows[flow_key]
        flow.observed_packet_count += 1
        if self.max_packets_per_flow is not None and len(flow.packets) >= self.max_packets_per_flow:
            flow.end_time = timestamp
            self._update_tcp_close_state(
                lookup_key,
                flow_key,
                direction,
                tcp_flags,
                active_flow_key,
                forward_endpoint,
                tcp_fin_directions,
            )
            return

        flow.packets.append(
            PacketViews(
                timestamp=timestamp,
                direction=direction,
                payload_only=payload,
                full_packet=bytes(packet),
                masked_header_packet=_mask_packet(packet),
                tcp_flags=tcp_flags,
            )
        )
        flow.end_time = timestamp
        self._update_tcp_close_state(
            lookup_key,
            flow_key,
            direction,
            tcp_flags,
            active_flow_key,
            forward_endpoint,
            tcp_fin_directions,
        )


def _raw_metadata_timestamp(metadata: Any) -> float:
    if hasattr(metadata, "sec"):
        return float(metadata.sec) + float(metadata.usec) / 1_000_000
    return float(((metadata.tshigh << 32) + metadata.tslow) / metadata.tsresol)
