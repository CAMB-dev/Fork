from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_bert.config import write_json


CONTEXT_COLUMNS = [
    "context_host_prev_60s_count",
    "context_host_prev_300s_count",
    "context_host_prev_60s_reset_count",
    "context_host_prev_60s_control_count",
    "context_host_prev_60s_payloadless_count",
    "context_host_prev_60s_unique_dst_ports",
    "context_host_prev_60s_unique_dst_hosts",
]


def _host(endpoint: Any) -> str:
    text = str(endpoint)
    if ":" not in text:
        return text
    return text.rsplit(":", 1)[0]


def _port(endpoint: Any) -> str:
    text = str(endpoint)
    if ":" not in text:
        return ""
    return text.rsplit(":", 1)[1]


def _compute_group_context(group: pd.DataFrame) -> dict[str, list[int]]:
    rows = list(
        group[
            [
                "_original_index",
                "start_time",
                "connection_type",
                "payload_byte_length",
                "_dst_host",
                "_dst_port",
            ]
        ].itertuples(index=False, name=None)
    )
    rows.sort(key=lambda item: (float(item[1]), int(item[0])))
    q60: deque[tuple] = deque()
    q300: deque[tuple] = deque()
    dst_host_counts: Counter[str] = Counter()
    dst_port_counts: Counter[str] = Counter()
    reset_count = 0
    control_count = 0
    payloadless_count = 0
    output = {column: [] for column in CONTEXT_COLUMNS}
    indexes: list[int] = []

    def remove60(item: tuple) -> None:
        nonlocal reset_count, control_count, payloadless_count
        _, _, connection_type, payload_byte_length, dst_host, dst_port = item
        if str(connection_type) == "tcp_reset_or_refused":
            reset_count -= 1
        if str(connection_type) == "tcp_control_only":
            control_count -= 1
        if int(payload_byte_length or 0) <= 0:
            payloadless_count -= 1
        dst_host_counts[str(dst_host)] -= 1
        if dst_host_counts[str(dst_host)] <= 0:
            del dst_host_counts[str(dst_host)]
        dst_port_counts[str(dst_port)] -= 1
        if dst_port_counts[str(dst_port)] <= 0:
            del dst_port_counts[str(dst_port)]

    for item in rows:
        index, start_time, connection_type, payload_byte_length, dst_host, dst_port = item
        start_time = float(start_time)
        while q60 and start_time - float(q60[0][1]) > 60.0:
            remove60(q60.popleft())
        while q300 and start_time - float(q300[0][1]) > 300.0:
            q300.popleft()

        indexes.append(int(index))
        output["context_host_prev_60s_count"].append(len(q60))
        output["context_host_prev_300s_count"].append(len(q300))
        output["context_host_prev_60s_reset_count"].append(max(reset_count, 0))
        output["context_host_prev_60s_control_count"].append(max(control_count, 0))
        output["context_host_prev_60s_payloadless_count"].append(max(payloadless_count, 0))
        output["context_host_prev_60s_unique_dst_ports"].append(len(dst_port_counts))
        output["context_host_prev_60s_unique_dst_hosts"].append(len(dst_host_counts))

        q60.append(item)
        q300.append(item)
        if str(connection_type) == "tcp_reset_or_refused":
            reset_count += 1
        if str(connection_type) == "tcp_control_only":
            control_count += 1
        if int(payload_byte_length or 0) <= 0:
            payloadless_count += 1
        dst_host_counts[str(dst_host)] += 1
        dst_port_counts[str(dst_port)] += 1

    output["_original_index"] = indexes
    return output


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"endpoint_a", "endpoint_b", "start_time", "connection_type", "payload_byte_length"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing required context columns: {sorted(missing)}")
    work = frame.copy()
    for column in CONTEXT_COLUMNS:
        work[column] = 0
    work["_original_index"] = range(len(work))
    work["_host"] = work["endpoint_a"].map(_host)
    work["_dst_host"] = work["endpoint_b"].map(_host)
    work["_dst_port"] = work["endpoint_b"].map(_port)
    context_frames = []
    group_columns = ["source_file", "_host"] if "source_file" in work.columns else ["_host"]
    for _, group in work.groupby(group_columns, sort=False):
        context_frames.append(pd.DataFrame(_compute_group_context(group)))
    if context_frames:
        context = pd.concat(context_frames, ignore_index=True).set_index("_original_index")
        for column in CONTEXT_COLUMNS:
            work.loc[context.index, column] = context[column].astype("int32")
    return work.drop(columns=["_original_index", "_host", "_dst_host", "_dst_port"])


def process_file(input_path: Path, output_path: Path) -> dict:
    frame = pd.read_parquet(input_path)
    output = add_context(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(output, preserve_index=False), output_path, compression="zstd")
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(output)),
        "context_columns": CONTEXT_COLUMNS,
        "max_values": {column: int(output[column].max()) if len(output) else 0 for column in CONTEXT_COLUMNS},
    }
    write_json(output_path.with_suffix(".context.json"), stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add causal host-window context columns to split parquet files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = {}
    for split in ["train", "val", "test"]:
        input_path = args.input_dir / f"{split}.parquet"
        output_path = args.output_dir / f"{split}.parquet"
        summaries[split] = process_file(input_path, output_path)
        for suffix in [".validate.json", ".class_distribution.csv"]:
            sidecar = args.input_dir / f"{split}{suffix}"
            if sidecar.exists():
                target = args.output_dir / sidecar.name
                target.write_bytes(sidecar.read_bytes())
    stats_path = args.input_dir / "split.stats.json"
    if stats_path.exists():
        (args.output_dir / "split.stats.json").write_bytes(stats_path.read_bytes())
    write_json(args.output_dir / "context.stats.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
