"""Helpers for aligning CIC-style flow CSV labels with reconstructed flows."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
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
    proto = str(protocol).strip().lower()
    try:
        proto_number = int(float(proto))
    except ValueError:
        proto_number = None
    if proto in {"tcp"} or proto_number == 6:
        proto = "tcp"
    elif proto in {"udp"} or proto_number == 17:
        proto = "udp"
    left = f"{src_ip}:{int(src_port)}"
    right = f"{dst_ip}:{int(dst_port)}"
    a, b = sorted([left, right])
    return proto, a, b


def directional_flow_key(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: str | int,
) -> tuple[str, str, str]:
    proto = str(protocol).strip().lower()
    try:
        proto_number = int(float(proto))
    except ValueError:
        proto_number = None
    if proto in {"tcp"} or proto_number == 6:
        proto = "tcp"
    elif proto in {"udp"} or proto_number == 17:
        proto = "udp"
    return proto, f"{src_ip}:{int(src_port)}", f"{dst_ip}:{int(dst_port)}"


def parse_cic_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    text = str(value).strip()
    if len(text) >= 10 and text[:4].isdigit() and text[4] in {"-", "/"}:
        parsed = pd.to_datetime(text, errors="coerce", yearfirst=True)
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    for dayfirst in (True, False):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
        if not pd.isna(parsed):
            return pd.Timestamp(parsed)
    return None


def normalize_lookup_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo is not None else value
    if isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if isinstance(value, int | float):
        return pd.Timestamp.fromtimestamp(float(value), tz=timezone.utc).tz_localize(None)
    return parse_cic_timestamp(value)


@dataclass(frozen=True)
class CicLabelRecord:
    key: tuple[str, str, str]
    directional_key: tuple[str, str, str]
    label: str
    timestamp: pd.Timestamp | None = None


@dataclass(frozen=True)
class CicLabelMatch:
    label: str
    mode: str
    time_delta_seconds: float | None = None


class CicFlowLabelIndex:
    """In-memory lookup for CIC flow label CSV rows."""

    def __init__(self, records: list[CicLabelRecord]) -> None:
        self.by_key: dict[tuple[str, str, str], list[CicLabelRecord]] = {}
        self.by_directional_key: dict[tuple[str, str, str], list[CicLabelRecord]] = {}
        for record in records:
            self.by_key.setdefault(record.key, []).append(record)
            self.by_directional_key.setdefault(record.directional_key, []).append(record)
        self.by_key_timed = self._build_timed_index(self.by_key)
        self.by_directional_key_timed = self._build_timed_index(self.by_directional_key)

    @staticmethod
    def _build_timed_index(
        index: dict[tuple[str, str, str], list[CicLabelRecord]],
    ) -> dict[tuple[str, str, str], tuple[list[pd.Timestamp], list[CicLabelRecord]]]:
        timed_index: dict[
            tuple[str, str, str],
            tuple[list[pd.Timestamp], list[CicLabelRecord]],
        ] = {}
        for key, records in index.items():
            timed = sorted(
                (record for record in records if record.timestamp is not None),
                key=lambda record: record.timestamp,
            )
            if timed:
                timed_index[key] = (
                    [record.timestamp for record in timed if record.timestamp is not None],
                    timed,
                )
        return timed_index

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "CicFlowLabelIndex":
        frame = normalize_cic_columns(frame)
        source_ip_column = "source_ip" if "source_ip" in frame else "src_ip"
        destination_ip_column = (
            "destination_ip" if "destination_ip" in frame else "dst_ip"
        )
        source_port_column = "source_port" if "source_port" in frame else "src_port"
        destination_port_column = (
            "destination_port" if "destination_port" in frame else "dst_port"
        )
        required_columns = [
            source_ip_column,
            destination_ip_column,
            source_port_column,
            destination_port_column,
            "label",
        ]
        missing = [column for column in required_columns if column not in frame]
        if missing:
            raise ValueError(f"missing CIC label columns: {', '.join(missing)}")

        selected_columns = required_columns + [
            "protocol" if "protocol" in frame else None,
            "timestamp" if "timestamp" in frame else None,
        ]
        selected_columns = [column for column in selected_columns if column is not None]
        work = frame[selected_columns].dropna(subset=required_columns).copy()
        work["label"] = work["label"].astype(str).str.strip()
        work = work[work["label"] != ""]
        if "protocol" not in work:
            work["protocol"] = ""
        if "timestamp" in work:
            timestamps = work["timestamp"].map(parse_cic_timestamp)
        else:
            timestamps = pd.Series([pd.NaT] * len(work), index=work.index)

        records: list[CicLabelRecord] = []
        for row, timestamp_value in zip(
            work.itertuples(index=False),
            timestamps,
            strict=True,
        ):
            data = row._asdict()
            timestamp = normalize_lookup_timestamp(timestamp_value)
            records.append(
                CicLabelRecord(
                    key=canonical_flow_key(
                        data[source_ip_column],
                        data[destination_ip_column],
                        int(data[source_port_column]),
                        int(data[destination_port_column]),
                        data["protocol"],
                    ),
                    directional_key=directional_flow_key(
                        data[source_ip_column],
                        data[destination_ip_column],
                        int(data[source_port_column]),
                        int(data[destination_port_column]),
                        data["protocol"],
                    ),
                    label=str(data["label"]),
                    timestamp=timestamp,
                )
            )
        return cls(records)

    def _best_match(
        self,
        candidates: list[CicLabelRecord],
        lookup_timestamp: pd.Timestamp | None,
        max_time_delta_seconds: float | None,
        timed_candidates: tuple[list[pd.Timestamp], list[CicLabelRecord]] | None = None,
    ) -> tuple[CicLabelRecord | None, float | None]:
        if not candidates:
            return None, None
        if lookup_timestamp is None:
            return candidates[0], None
        if timed_candidates is None:
            return candidates[0], None
        timestamps, records = timed_candidates
        if not timestamps:
            return candidates[0], None
        insert_at = bisect_left(timestamps, lookup_timestamp)
        nearest: list[CicLabelRecord] = []
        if insert_at < len(records):
            nearest.append(records[insert_at])
        if insert_at > 0:
            nearest.append(records[insert_at - 1])
        best = min(nearest, key=lambda item: abs(item.timestamp - lookup_timestamp))
        delta = abs(best.timestamp - lookup_timestamp).total_seconds()
        if max_time_delta_seconds is not None and delta > max_time_delta_seconds:
            return None, delta
        return best, delta

    def lookup_detailed(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str | int,
        timestamp: Any = None,
        max_time_delta_seconds: float | None = None,
    ) -> CicLabelMatch | None:
        lookup_timestamp = normalize_lookup_timestamp(timestamp)
        exact_key = directional_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
        reverse_key = directional_flow_key(dst_ip, src_ip, dst_port, src_port, protocol)
        canonical_key = canonical_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)

        directed_matches: list[tuple[str, CicLabelRecord, float | None]] = []
        for mode, key, candidates in [
            ("directional", exact_key, self.by_directional_key.get(exact_key, [])),
            (
                "directional_reversed",
                reverse_key,
                self.by_directional_key.get(reverse_key, []),
            ),
        ]:
            best, delta = self._best_match(
                candidates,
                lookup_timestamp,
                max_time_delta_seconds,
                self.by_directional_key_timed.get(key),
            )
            if best is not None:
                directed_matches.append((mode, best, delta))
        if directed_matches:
            if lookup_timestamp is None:
                mode, best, delta = directed_matches[0]
            else:
                mode, best, delta = min(
                    directed_matches,
                    key=lambda item: (
                        float("inf") if item[2] is None else item[2],
                        0 if item[0] == "directional" else 1,
                    ),
                )
            return CicLabelMatch(best.label, mode, delta)

        best, delta = self._best_match(
            self.by_key.get(canonical_key, []),
            lookup_timestamp,
            max_time_delta_seconds,
            self.by_key_timed.get(canonical_key),
        )
        if best is not None:
            return CicLabelMatch(best.label, "canonical_fallback", delta)
        return None

    def lookup(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str | int,
        timestamp: Any = None,
        max_time_delta_seconds: float | None = None,
    ) -> str | None:
        match = self.lookup_detailed(
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
            timestamp=timestamp,
            max_time_delta_seconds=max_time_delta_seconds,
        )
        return None if match is None else match.label
