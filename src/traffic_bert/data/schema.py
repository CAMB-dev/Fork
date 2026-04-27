"""Common data structures used by the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class InputView(str, Enum):
    PAYLOAD_ONLY = "payload_only"
    FULL_PACKET = "full_packet"
    MASKED_HEADER_PACKET = "masked_header_packet"


@dataclass(frozen=True)
class PacketViews:
    timestamp: float
    direction: str
    payload_only: bytes
    full_packet: bytes
    masked_header_packet: bytes

    def bytes_for_view(self, view: InputView) -> bytes:
        if view == InputView.PAYLOAD_ONLY:
            return self.payload_only
        if view == InputView.FULL_PACKET:
            return self.full_packet
        if view == InputView.MASKED_HEADER_PACKET:
            return self.masked_header_packet
        raise ValueError(f"unsupported input view: {view}")


@dataclass
class FlowRecord:
    flow_id: str
    source_file: str
    protocol: str
    endpoint_a: str
    endpoint_b: str
    start_time: float
    end_time: float
    packets: list[PacketViews]

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def payload_byte_length(self) -> int:
        return sum(len(packet.payload_only) for packet in self.packets)

    @property
    def packet_byte_length(self) -> int:
        return sum(len(packet.full_packet) for packet in self.packets)

    @property
    def has_payload(self) -> bool:
        return self.payload_byte_length > 0

    def to_row(
        self,
        view: InputView,
        source_dataset: str,
        source_label: str,
        major_label: str,
        minor_labels: list[str],
        split: str,
    ) -> dict[str, Any]:
        packet_bytes = [packet.bytes_for_view(view) for packet in self.packets]
        return {
            "flow_id": self.flow_id,
            "source_dataset": source_dataset,
            "source_file": self.source_file,
            "source_label": source_label,
            "major_label": major_label,
            "minor_labels": minor_labels,
            "split": split,
            "view": view.value,
            "protocol": self.protocol,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "packet_count": self.packet_count,
            "payload_byte_length": self.payload_byte_length,
            "packet_byte_length": self.packet_byte_length,
            "has_payload": self.has_payload,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "packet_directions": [packet.direction for packet in self.packets],
            "packet_lengths": [len(data) for data in packet_bytes],
            "bytes": b"".join(packet_bytes),
        }


def collect_pcap_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(
        item
        for item in root.rglob("*")
        if item.suffix.lower() in {".pcap", ".pcapng"}
    )
