"""Diagnose CICIDS CSV-to-PCAP flow alignment."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _diagnose_flows(
    *,
    pcap_path: Path,
    csv_name: str,
    frame: pd.DataFrame,
    flows: list,
    max_time_delta_seconds: float | None,
) -> dict:
    csv_keys, csv_attack_keys, csv_label_counts = _csv_key_counts(frame)
    label_index = CicFlowLabelIndex.from_frame(frame)

    pcap_key_counts = Counter(_flow_key(flow) for flow in flows)
    pcap_keys = set(pcap_key_counts)
    csv_key_set = set(csv_keys)
    attack_key_set = set(csv_attack_keys)

    matched_labels = Counter()
    label_match_modes = Counter()
    unmatched_flows = 0
    payload_by_label = Counter()
    empty_payload_by_label = Counter()
    for flow in flows:
        src_ip, src_port = _endpoint_parts(flow.endpoint_a)
        dst_ip, dst_port = _endpoint_parts(flow.endpoint_b)
        match = label_index.lookup_detailed(
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            flow.protocol,
            timestamp=flow.start_time,
            max_time_delta_seconds=max_time_delta_seconds,
        )
        if match is None:
            unmatched_flows += 1
            continue
        label = match.label
        matched_labels[str(label)] += 1
        label_match_modes[match.mode] += 1
        if flow.has_payload:
            payload_by_label[str(label)] += 1
        else:
            empty_payload_by_label[str(label)] += 1

    return {
        "pcap_path": str(pcap_path),
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
        "label_match_modes": dict(label_match_modes),
        "payload_matched_labels": dict(payload_by_label),
        "empty_payload_matched_labels": dict(empty_payload_by_label),
        "unmatched_pcap_flows": unmatched_flows,
        "cic_label_max_time_delta_seconds": max_time_delta_seconds,
    }


def _extract_flows(payload: dict) -> list:
    return PcapFlowExtractor(
        max_packets_per_flow=payload["max_packets_per_flow"],
        max_packets_to_read=payload["max_packets_to_read"],
        max_packets_to_skip=payload["max_packets_to_skip"],
        min_packet_time=payload["min_packet_time"],
        max_packet_time=payload["max_packet_time"],
        flow_timeout_seconds=payload["flow_timeout_seconds"],
    ).extract(payload["pcap_path"])


def _read_build_window(build_json: Path | None) -> tuple[float | None, float | None]:
    if build_json is None or not build_json.exists():
        return None, None
    try:
        payload = json.loads(build_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return payload.get("min_packet_time"), payload.get("max_packet_time")


def _payload_extract_key(payload: dict) -> tuple:
    return (
        str(payload["pcap_path"]),
        payload["max_packets_per_flow"],
        payload["max_packets_to_read"],
        payload["max_packets_to_skip"],
        payload["min_packet_time"],
        payload["max_packet_time"],
        payload["flow_timeout_seconds"],
    )


def _run_payload_group(payloads: list[dict]) -> list[dict]:
    first = payloads[0]
    flows = _extract_flows(first)
    results = []
    for payload in payloads:
        frame, csv_name = _read_csv(
            payload["label_zip"],
            payload["csv_path"],
            tuple(payload.get("csv_contains", ())),
        )
        result = _diagnose_flows(
            pcap_path=payload["pcap_path"],
            csv_name=csv_name,
            frame=frame,
            flows=flows,
            max_time_delta_seconds=payload["cic_label_max_time_delta_seconds"],
        )
        _, csv_attack_keys, _ = _csv_key_counts(frame)
        pcap_keys = set(_flow_key(flow) for flow in flows)
        result.update(
            {
                "slug": payload["slug"],
                "min_packet_time": payload["min_packet_time"],
                "max_packet_time": payload["max_packet_time"],
                "flow_timeout_seconds": payload["flow_timeout_seconds"],
                "sample_missing_attack_keys": [
                    list(item)
                    for item in list(set(csv_attack_keys) - pcap_keys)[
                        : payload["sample_size"]
                    ]
                ],
            }
        )
        results.append(result)
    return results


def _job_payloads(args: argparse.Namespace) -> list[dict]:
    raw_jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))
    payloads = []
    for raw in raw_jobs:
        if args.no_time_window:
            min_packet_time, max_packet_time = None, None
        elif "min_packet_time" in raw or "max_packet_time" in raw:
            min_packet_time = raw.get("min_packet_time")
            max_packet_time = raw.get("max_packet_time")
        else:
            build_json = Path(raw["build_json"]) if raw.get("build_json") else None
            min_packet_time, max_packet_time = _read_build_window(build_json)
        payloads.append(
            {
                "slug": raw["slug"],
                "pcap_path": Path(raw["pcap_path"]),
                "label_zip": Path(raw["label_zip"]) if raw.get("label_zip") else None,
                "csv_path": Path(raw["csv_path"]) if raw.get("csv_path") else None,
                "csv_contains": tuple(raw.get("csv_contains", ())),
                "max_packets_to_read": args.max_packets_to_read,
                "max_packets_to_skip": args.max_packets_to_skip,
                "max_packets_per_flow": args.max_packets_per_flow,
                "min_packet_time": min_packet_time,
                "max_packet_time": max_packet_time,
                "flow_timeout_seconds": args.flow_timeout_seconds,
                "cic_label_max_time_delta_seconds": args.cic_label_max_time_delta_seconds,
                "sample_size": args.sample_size,
            }
        )
    return payloads


def _write_grouped_outputs(results: list[dict], output_dir: Path, summary_output: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for result in sorted(results, key=lambda item: item["slug"]):
        slug = result["slug"]
        (output_dir / f"{slug}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[slug] = {
            "csv_label_counts": result.get("csv_label_counts", {}),
            "pcap_flows": result.get("pcap_flows"),
            "key_intersection": result.get("key_intersection"),
            "attack_key_intersection": result.get("attack_key_intersection"),
            "matched_labels_from_pcap_flows": result.get("matched_labels_from_pcap_flows", {}),
            "label_match_modes": result.get("label_match_modes", {}),
            "payload_matched_labels": result.get("payload_matched_labels", {}),
            "unmatched_pcap_flows": result.get("unmatched_pcap_flows"),
            "min_packet_time": result.get("min_packet_time"),
            "max_packet_time": result.get("max_packet_time"),
        }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap-path", type=Path)
    parser.add_argument("--label-zip", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--csv-contains", nargs="*", default=())
    parser.add_argument("--max-packets-to-read", type=int, default=None)
    parser.add_argument("--max-packets-to-skip", type=int, default=0)
    parser.add_argument("--max-packets-per-flow", type=int, default=16)
    parser.add_argument("--min-packet-time", type=float, default=None)
    parser.add_argument("--max-packet-time", type=float, default=None)
    parser.add_argument("--flow-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cic-label-max-time-delta-seconds", type=float, default=900.0)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--jobs-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cicids2017_alignment"))
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-time-window", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs_json is not None:
        payloads = _job_payloads(args)
        grouped_payloads: dict[tuple, list[dict]] = {}
        for payload in payloads:
            grouped_payloads.setdefault(_payload_extract_key(payload), []).append(payload)
        if len(grouped_payloads) < len(payloads):
            print(
                "Grouped CICIDS alignment diagnostics by PCAP extraction parameters: "
                f"{len(payloads)} jobs -> {len(grouped_payloads)} PCAP scans"
            )
        results = []
        groups = list(grouped_payloads.values())
        if args.max_workers <= 1:
            for group in groups:
                results.extend(_run_payload_group(group))
        else:
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(_run_payload_group, group) for group in groups]
                for future in as_completed(futures):
                    results.extend(future.result())
        summary_output = args.summary_output or (args.output_dir / "summary.json")
        _write_grouped_outputs(results, args.output_dir, summary_output)
        return

    if args.pcap_path is None:
        raise ValueError("--pcap-path is required unless --jobs-json is used")
    frame, csv_name = _read_csv(args.label_zip, args.csv_path, tuple(args.csv_contains))

    payload = {
        "pcap_path": args.pcap_path,
        "max_packets_per_flow": args.max_packets_per_flow,
        "max_packets_to_read": args.max_packets_to_read,
        "max_packets_to_skip": args.max_packets_to_skip,
        "min_packet_time": args.min_packet_time,
        "max_packet_time": args.max_packet_time,
        "flow_timeout_seconds": args.flow_timeout_seconds,
    }
    flows = _extract_flows(payload)
    result = _diagnose_flows(
        pcap_path=args.pcap_path,
        csv_name=csv_name,
        frame=frame,
        flows=flows,
        max_time_delta_seconds=args.cic_label_max_time_delta_seconds,
    )
    _, csv_attack_keys, _ = _csv_key_counts(frame)
    pcap_keys = set(_flow_key(flow) for flow in flows)
    missing_attack_keys = list(set(csv_attack_keys) - pcap_keys)[: args.sample_size]
    result["sample_missing_attack_keys"] = [list(item) for item in missing_attack_keys]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
