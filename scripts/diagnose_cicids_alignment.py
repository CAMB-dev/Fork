"""Diagnose CICIDS CSV-to-PCAP flow alignment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import pandas as pd

from traffic_bert.data.cic import (
    CicFlowLabelIndex,
    canonical_flow_key,
    normalize_cic_columns,
)
from traffic_bert.data.pcap import PcapFlowExtractor


def _find_csv(label_zip: Path, contains: tuple[str, ...]) -> str:
    needles = tuple(item.lower() for item in contains)
    with zipfile.ZipFile(label_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and all(needle in Path(name).name.lower() for needle in needles)
        ]
    if not matches:
        raise FileNotFoundError(f"no CSV matching {contains!r} in {label_zip}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous CSV match for {contains!r}: {matches}")
    return matches[0]


def _read_csv(
    label_zip: Path | None,
    csv_path: Path | None,
    contains: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    if csv_path is not None:
        return pd.read_csv(csv_path, encoding="latin1", low_memory=False), str(csv_path)
    if label_zip is None:
        raise ValueError("--label-zip or --csv-path is required")
    name = _find_csv(label_zip, contains)
    with zipfile.ZipFile(label_zip) as archive:
        with archive.open(name) as handle:
            return pd.read_csv(handle, encoding="latin1", low_memory=False), name


def _endpoint_parts(endpoint: str) -> tuple[str, int]:
    host, port = endpoint.rsplit(":", 1)
    return host, int(port)


def _flow_key(flow) -> tuple[str, str, str]:
    src_ip, src_port = _endpoint_parts(flow.endpoint_a)
    dst_ip, dst_port = _endpoint_parts(flow.endpoint_b)
    return canonical_flow_key(src_ip, dst_ip, src_port, dst_port, flow.protocol)


def _csv_key_counts(frame: pd.DataFrame) -> tuple[Counter, Counter, Counter]:
    work = normalize_cic_columns(frame)
    source_ip_column = "source_ip" if "source_ip" in work else "src_ip"
    destination_ip_column = "destination_ip" if "destination_ip" in work else "dst_ip"
    source_port_column = "source_port" if "source_port" in work else "src_port"
    destination_port_column = "destination_port" if "destination_port" in work else "dst_port"
    protocol_column = "protocol" if "protocol" in work else None
    required = [
        source_ip_column,
        destination_ip_column,
        source_port_column,
        destination_port_column,
        "label",
    ]
    missing = [column for column in required if column not in work]
    if missing:
        raise ValueError(f"missing CIC columns: {missing}")

    key_counts: Counter = Counter()
    attack_key_counts: Counter = Counter()
    label_counts: Counter = Counter()
    for row in work.dropna(subset=required).itertuples(index=False):
        data = row._asdict()
        label = str(data["label"])
        protocol = data[protocol_column] if protocol_column else ""
        key = canonical_flow_key(
            data[source_ip_column],
            data[destination_ip_column],
            int(data[source_port_column]),
            int(data[destination_port_column]),
            protocol,
        )
        key_counts[key] += 1
        label_counts[label] += 1
        if label.upper() != "BENIGN":
            attack_key_counts[key] += 1
    return key_counts, attack_key_counts, label_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap-path", type=Path, required=True)
    parser.add_argument("--label-zip", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--csv-contains", nargs="*", default=())
    parser.add_argument("--max-packets-to-read", type=int, default=None)
    parser.add_argument("--max-packets-to-skip", type=int, default=0)
    parser.add_argument("--max-packets-per-flow", type=int, default=16)
    parser.add_argument("--min-packet-time", type=float, default=None)
    parser.add_argument("--max-packet-time", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, csv_name = _read_csv(args.label_zip, args.csv_path, tuple(args.csv_contains))
    csv_keys, csv_attack_keys, csv_label_counts = _csv_key_counts(frame)
    label_index = CicFlowLabelIndex.from_frame(frame)

    flows = PcapFlowExtractor(
        max_packets_per_flow=args.max_packets_per_flow,
        max_packets_to_read=args.max_packets_to_read,
        max_packets_to_skip=args.max_packets_to_skip,
        min_packet_time=args.min_packet_time,
        max_packet_time=args.max_packet_time,
    ).extract(args.pcap_path)
    pcap_key_counts = Counter(_flow_key(flow) for flow in flows)
    pcap_keys = set(pcap_key_counts)
    csv_key_set = set(csv_keys)
    attack_key_set = set(csv_attack_keys)

    matched_labels = Counter()
    unmatched_flows = 0
    payload_by_label = Counter()
    empty_payload_by_label = Counter()
    for flow in flows:
        src_ip, src_port = _endpoint_parts(flow.endpoint_a)
        dst_ip, dst_port = _endpoint_parts(flow.endpoint_b)
        label = label_index.lookup(
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            flow.protocol,
            timestamp=flow.start_time,
        )
        if label is None:
            unmatched_flows += 1
            continue
        matched_labels[str(label)] += 1
        if flow.has_payload:
            payload_by_label[str(label)] += 1
        else:
            empty_payload_by_label[str(label)] += 1

    missing_attack_keys = list(attack_key_set - pcap_keys)[: args.sample_size]
    result = {
        "pcap_path": str(args.pcap_path),
        "csv_name": csv_name,
        "csv_rows": int(len(frame)),
        "csv_label_counts": dict(csv_label_counts),
        "csv_unique_keys": len(csv_key_set),
        "csv_attack_unique_keys": len(attack_key_set),
        "pcap_flows": len(flows),
        "pcap_unique_keys": len(pcap_keys),
        "key_intersection": len(csv_key_set & pcap_keys),
        "attack_key_intersection": len(attack_key_set & pcap_keys),
        "matched_labels_from_pcap_flows": dict(matched_labels),
        "payload_matched_labels": dict(payload_by_label),
        "empty_payload_matched_labels": dict(empty_payload_by_label),
        "unmatched_pcap_flows": unmatched_flows,
        "sample_missing_attack_keys": [list(item) for item in missing_attack_keys],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
