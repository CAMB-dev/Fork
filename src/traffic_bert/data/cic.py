"""Helpers for aligning CIC-style flow CSV labels with reconstructed flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


def normalize_column_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def normalize_cic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: normalize_column_name(column) for column in frame.columns})


def canonical_flow_key(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: str | int,
) -> tuple[str, str, str]:
    proto = str(protocol).lower()
    if proto in {"6", "tcp"}:
        proto = "tcp"
    elif proto in {"17", "udp"}:
        proto = "udp"
    left = f"{src_ip}:{int(src_port)}"
    right = f"{dst_ip}:{int(dst_port)}"
    a, b = sorted([left, right])
    return proto, a, b


def parse_cic_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    text = str(value).strip()
    for dayfirst in (False, True):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    return None


@dataclass(frozen=True)
class CicLabelRecord:
    key: tuple[str, str, str]
    label: str
    timestamp: pd.Timestamp | None = None


class CicFlowLabelIndex:
    """In-memory lookup for CIC flow label CSV rows."""

    def __init__(self, records: list[CicLabelRecord]) -> None:
        self.by_key: dict[tuple[str, str, str], list[CicLabelRecord]] = {}
        for record in records:
            self.by_key.setdefault(record.key, []).append(record)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CicFlowLabelIndex":
        frame = normalize_cic_columns(frame)
        records: list[CicLabelRecord] = []
        for _, row in frame.iterrows():
            src_ip = row.get("source_ip") or row.get("src_ip")
            dst_ip = row.get("destination_ip") or row.get("dst_ip")
            src_port = row.get("source_port") or row.get("src_port")
            dst_port = row.get("destination_port") or row.get("dst_port")
            protocol = row.get("protocol", "")
            label = row.get("label")
            if pd.isna(src_ip) or pd.isna(dst_ip) or pd.isna(src_port) or pd.isna(dst_port):
                continue
            if label is None or pd.isna(label):
                continue
            timestamp = parse_cic_timestamp(row.get("timestamp"))
            records.append(
                CicLabelRecord(
                    key=canonical_flow_key(src_ip, dst_ip, int(src_port), int(dst_port), protocol),
                    label=str(label),
                    timestamp=timestamp,
                )
            )
        return cls(records)

    def lookup(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str | int,
        timestamp: pd.Timestamp | None = None,
    ) -> str | None:
        key = canonical_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        candidates = self.by_key.get(key, [])
        if not candidates:
            return None
        if timestamp is None or len(candidates) == 1:
            return candidates[0].label
        candidates_with_time = [item for item in candidates if item.timestamp is not None]
        if not candidates_with_time:
            return candidates[0].label
        best = min(candidates_with_time, key=lambda item: abs(item.timestamp - timestamp))
        return best.label

