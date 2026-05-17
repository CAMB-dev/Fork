"""Build processed Parquet datasets from raw PCAP files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from traffic_bert.config import load_yaml
from traffic_bert.data.cic import CicFlowLabelIndex, CicLabelMatch
from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.schema import FlowRecord, InputView, collect_pcap_files
from traffic_bert.labels import LabelMap


@dataclass(frozen=True)
class BuildConfig:
    input_path: Path
    output_path: Path
    label_map_path: Path
    source_dataset: str
    split: str = "train"
    label_source: str = "filename"
    static_label: str | None = None
    label_csv_path: Path | None = None
    drop_unmatched_labels: bool = True
    views: tuple[InputView, ...] = (
        InputView.PAYLOAD_ONLY,
        InputView.FULL_PACKET,
        InputView.MASKED_HEADER_PACKET,
    )
    keep_empty_payload: bool = True
    max_packets_per_flow: int | None = None
    max_packets_to_read: int | None = None
    max_packets_to_skip: int = 0
    min_packet_time: float | None = None
    max_packet_time: float | None = None
    flow_timeout_seconds: float | None = 120.0
    close_on_tcp_flags: bool = False
    cic_label_max_time_delta_seconds: float | None = None


def infer_source_label(path: Path, label_source: str, static_label: str | None = None) -> str:
    if label_source == "static":
        if not static_label:
            raise ValueError("static label_source requires static_label")
        return static_label
    if label_source == "parent":
        return path.parent.name
    if label_source == "filename":
        return path.stem
    if label_source == "parent_filename":
        return f"{path.parent.name}/{path.stem}"
    if label_source == "cic_csv":
        raise ValueError("cic_csv labels are resolved per flow, not per file")
    raise ValueError(f"unsupported label_source: {label_source}")


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, port = endpoint.rsplit(":", 1)
    return host, int(port)


def _label_match_for_flow(
    config: BuildConfig,
    pcap_path: Path,
    flow_initiator_endpoint: str,
    flow_responder_endpoint: str,
    protocol: str,
    start_time: float,
    cic_index: CicFlowLabelIndex | None,
) -> CicLabelMatch | None:
    if config.label_source == "cic_csv":
        if cic_index is None:
            raise ValueError("label_source=cic_csv requires label_csv_path")
        src_ip, src_port = _parse_endpoint(flow_initiator_endpoint)
        dst_ip, dst_port = _parse_endpoint(flow_responder_endpoint)
        return cic_index.lookup_detailed(
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
            timestamp=start_time,
            max_time_delta_seconds=config.cic_label_max_time_delta_seconds,
        )
    return CicLabelMatch(
        infer_source_label(pcap_path, config.label_source, config.static_label),
        config.label_source,
        None,
    )


def build_config_from_dict(raw: dict) -> list[BuildConfig]:
    """Parse one or more BuildConfig entries from a YAML-ready mapping."""

    common = raw.get("build", {})
    entries = raw.get("datasets")
    if entries is None:
        entries = [raw]

    configs: list[BuildConfig] = []
    for item in entries:
        merged = {**common, **item}
        views = tuple(
            InputView(value)
            for value in merged.get(
                "views",
                ["payload_only", "full_packet", "masked_header_packet"],
            )
        )
        configs.append(
            BuildConfig(
                input_path=Path(merged.get("input_path") or merged["raw_path"]),
                output_path=Path(merged.get("output_path") or merged["processed_path"]),
                label_map_path=Path(merged.get("label_map", raw.get("label_map", "configs/label_map.yaml"))),
                source_dataset=merged.get("source_dataset", merged.get("dataset", "custom")),
                split=merged.get("split", "train"),
                label_source=merged.get("label_source", "filename"),
                static_label=merged.get("static_label"),
                label_csv_path=Path(merged["label_csv_path"])
                if merged.get("label_csv_path")
                else None,
                drop_unmatched_labels=bool(merged.get("drop_unmatched_labels", True)),
                views=views,
                keep_empty_payload=bool(merged.get("keep_empty_payload", True)),
                max_packets_per_flow=merged.get("max_packets_per_flow"),
                max_packets_to_read=merged.get("max_packets_to_read"),
                max_packets_to_skip=int(merged.get("max_packets_to_skip", 0)),
                min_packet_time=merged.get("min_packet_time"),
                max_packet_time=merged.get("max_packet_time"),
                flow_timeout_seconds=merged.get("flow_timeout_seconds", 120.0),
                close_on_tcp_flags=bool(merged.get("close_on_tcp_flags", False)),
                cic_label_max_time_delta_seconds=merged.get(
                    "cic_label_max_time_delta_seconds"
                ),
            )
        )
    return configs


def build_config_from_yaml(path: str | Path) -> list[BuildConfig]:
    return build_config_from_dict(load_yaml(path))


def build_processed_dataset(config: BuildConfig) -> dict:
    extractor = PcapFlowExtractor(
        max_packets_per_flow=config.max_packets_per_flow,
        max_packets_to_read=config.max_packets_to_read,
        max_packets_to_skip=config.max_packets_to_skip,
        min_packet_time=config.min_packet_time,
        max_packet_time=config.max_packet_time,
        flow_timeout_seconds=config.flow_timeout_seconds,
        close_on_tcp_flags=config.close_on_tcp_flags,
    )
    raw_files = collect_pcap_files(config.input_path)
    flows_by_file = {pcap_path: extractor.extract(pcap_path) for pcap_path in raw_files}
    return build_processed_dataset_from_flows(config, flows_by_file)


def build_processed_dataset_from_flows(
    config: BuildConfig,
    flows_by_file: dict[Path, list[FlowRecord]],
) -> dict:
    label_map = LabelMap.from_yaml(config.label_map_path)
    cic_index = None
    if config.label_source == "cic_csv":
        if config.label_csv_path is None:
            raise ValueError("label_source=cic_csv requires label_csv_path")
        cic_index = CicFlowLabelIndex.from_frame(
            pd.read_csv(config.label_csv_path, low_memory=False)
        )

    rows: list[dict] = []
    raw_files = list(flows_by_file)
    flow_counters = Counter(
        {
            "extracted_flows_total": 0,
            "matched_flows": 0,
            "unmatched_label_flows": 0,
            "empty_payload_flows": 0,
            "dropped_unmatched_label_flows": 0,
            "dropped_empty_payload_flows": 0,
        }
    )
    label_match_modes: Counter = Counter()
    for pcap_path, flows in flows_by_file.items():
        flow_counters["extracted_flows_total"] += len(flows)

        for flow in flows:
            if flow.was_packet_truncated:
                flow_counters["flows_at_packet_cap"] += 1
            label_match = _label_match_for_flow(
                config=config,
                pcap_path=pcap_path,
                flow_initiator_endpoint=flow.initiator_endpoint,
                flow_responder_endpoint=flow.responder_endpoint,
                protocol=flow.protocol,
                start_time=flow.start_time,
                cic_index=cic_index,
            )
            if label_match is None:
                flow_counters["unmatched_label_flows"] += 1
                if config.drop_unmatched_labels:
                    flow_counters["dropped_unmatched_label_flows"] += 1
                    continue
            else:
                flow_counters["matched_flows"] += 1
                label_match_modes[label_match.mode] += 1

            if not flow.has_payload:
                flow_counters["empty_payload_flows"] += 1
                if not config.keep_empty_payload:
                    flow_counters["dropped_empty_payload_flows"] += 1
                    continue

            source_label = "UNMATCHED" if label_match is None else label_match.label
            target = label_map.map_source_label(source_label)
            for view in config.views:
                rows.append(
                    flow.to_row(
                        view=view,
                        source_dataset=config.source_dataset,
                        source_label=source_label,
                        major_label=target.major_label,
                        minor_labels=list(target.minor_labels),
                        split=config.split,
                        label_match_mode=None if label_match is None else label_match.mode,
                        label_time_delta_seconds=None
                        if label_match is None
                        else label_match.time_delta_seconds,
                    )
                )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(config.output_path, index=False)
    stats = dataset_stats(
        frame,
        raw_files,
        {
            **flow_counters,
            "max_packets_per_flow": config.max_packets_per_flow,
            "max_packets_to_read": config.max_packets_to_read,
            "max_packets_to_skip": config.max_packets_to_skip,
            "min_packet_time": config.min_packet_time,
            "max_packet_time": config.max_packet_time,
            "flow_timeout_seconds": config.flow_timeout_seconds,
            "close_on_tcp_flags": config.close_on_tcp_flags,
            "cic_label_max_time_delta_seconds": config.cic_label_max_time_delta_seconds,
            "drop_unmatched_labels": config.drop_unmatched_labels,
            "keep_empty_payload": config.keep_empty_payload,
            "connection_type_version": 1,
            "tcp_close_policy_version": 3,
            "label_match_modes": dict(label_match_modes),
        },
    )

    stats_path = config.output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def dataset_stats(
    frame: pd.DataFrame,
    raw_files: list[Path],
    flow_counters: Counter | dict | None = None,
) -> dict:
    counters = {}
    for key, value in (flow_counters or {}).items():
        if isinstance(value, bool) or value is None:
            counters[str(key)] = value
        elif isinstance(value, int):
            counters[str(key)] = int(value)
        elif isinstance(value, float):
            counters[str(key)] = float(value)
        else:
            counters[str(key)] = value
    if frame.empty:
        return {
            "raw_files": [str(path) for path in raw_files],
            "rows": 0,
            "flows": 0,
            "views": {},
            "major_labels": {},
            "source_labels": {},
            "connection_types": {},
            **counters,
        }

    return {
        "raw_files": [str(path) for path in raw_files],
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()),
        "views": dict(Counter(frame["view"])),
        "major_labels": dict(Counter(frame["major_label"])),
        "source_labels": dict(Counter(frame["source_label"])),
        "connection_types": dict(Counter(frame["connection_type"]))
        if "connection_type" in frame
        else {},
        "packet_truncated_rows": int(frame["was_packet_truncated"].sum())
        if "was_packet_truncated" in frame
        else 0,
        "packet_truncated_flows": int(
            frame.loc[frame["was_packet_truncated"], "flow_id"].nunique()
        )
        if {"was_packet_truncated", "flow_id"} <= set(frame.columns)
        else 0,
        "empty_payload_rows": int((~frame["has_payload"]).sum()),
        **counters,
        "packet_count": {
            "min": int(frame["packet_count"].min()),
            "median": float(frame["packet_count"].median()),
            "max": int(frame["packet_count"].max()),
        },
        "byte_length": {
            "min": int(frame["packet_byte_length"].min()),
            "median": float(frame["packet_byte_length"].median()),
            "max": int(frame["packet_byte_length"].max()),
        },
    }
