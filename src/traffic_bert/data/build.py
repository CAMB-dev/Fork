"""Build processed Parquet datasets from raw PCAP files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.schema import InputView, collect_pcap_files
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
    views: tuple[InputView, ...] = (
        InputView.PAYLOAD_ONLY,
        InputView.FULL_PACKET,
        InputView.MASKED_HEADER_PACKET,
    )
    keep_empty_payload: bool = True
    max_packets_per_flow: int | None = None


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
    raise ValueError(f"unsupported label_source: {label_source}")


def build_processed_dataset(config: BuildConfig) -> dict:
    label_map = LabelMap.from_yaml(config.label_map_path)
    extractor = PcapFlowExtractor(max_packets_per_flow=config.max_packets_per_flow)

    rows: list[dict] = []
    raw_files = collect_pcap_files(config.input_path)
    for pcap_path in raw_files:
        source_label = infer_source_label(pcap_path, config.label_source, config.static_label)
        target = label_map.map_source_label(source_label)
        flows = extractor.extract(pcap_path)

        for flow in flows:
            if not config.keep_empty_payload and not flow.has_payload:
                continue
            for view in config.views:
                rows.append(
                    flow.to_row(
                        view=view,
                        source_dataset=config.source_dataset,
                        source_label=source_label,
                        major_label=target.major_label,
                        minor_labels=list(target.minor_labels),
                        split=config.split,
                    )
                )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(config.output_path, index=False)
    stats = dataset_stats(frame, raw_files)

    stats_path = config.output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def dataset_stats(frame: pd.DataFrame, raw_files: list[Path]) -> dict:
    if frame.empty:
        return {
            "raw_files": [str(path) for path in raw_files],
            "rows": 0,
            "flows": 0,
            "views": {},
            "major_labels": {},
            "source_labels": {},
        }

    return {
        "raw_files": [str(path) for path in raw_files],
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()),
        "views": dict(Counter(frame["view"])),
        "major_labels": dict(Counter(frame["major_label"])),
        "source_labels": dict(Counter(frame["source_label"])),
        "empty_payload_rows": int((~frame["has_payload"]).sum()),
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

