from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in Counter(frame[column].dropna()).items()}


def _read(path: Path) -> pd.DataFrame:
    columns = [
        "flow_id",
        "source_file",
        "source_label",
        "major_label",
        "minor_labels",
        "view",
        "packet_count",
        "payload_byte_length",
    ]
    return pd.read_parquet(path, columns=[column for column in columns if column])


def _split_summary(frame: pd.DataFrame) -> dict:
    source_files = set(str(value) for value in frame.get("source_file", pd.Series(dtype=str)).dropna())
    return {
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()) if "flow_id" in frame else int(len(frame)),
        "major_labels": _counts(frame, "major_label"),
        "source_labels": _counts(frame, "source_label"),
        "views": _counts(frame, "view"),
        "source_file_count": len(source_files),
        "packet_count": {
            "min": int(frame["packet_count"].min()) if "packet_count" in frame and len(frame) else None,
            "median": float(frame["packet_count"].median())
            if "packet_count" in frame and len(frame)
            else None,
            "max": int(frame["packet_count"].max()) if "packet_count" in frame and len(frame) else None,
        },
        "payload_byte_length": {
            "min": int(frame["payload_byte_length"].min())
            if "payload_byte_length" in frame and len(frame)
            else None,
            "median": float(frame["payload_byte_length"].median())
            if "payload_byte_length" in frame and len(frame)
            else None,
            "max": int(frame["payload_byte_length"].max())
            if "payload_byte_length" in frame and len(frame)
            else None,
        },
    }


def audit(train_path: Path, val_path: Path, test_path: Path) -> dict:
    frames = {
        "train": _read(train_path),
        "val": _read(val_path),
        "test": _read(test_path),
    }
    source_files = {
        split: set(str(value) for value in frame.get("source_file", pd.Series(dtype=str)).dropna())
        for split, frame in frames.items()
    }
    all_major_labels = sorted(
        {
            str(label)
            for frame in frames.values()
            for label in frame.get("major_label", pd.Series(dtype=str)).dropna().unique()
        }
    )
    overlap = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        values = sorted(source_files[left] & source_files[right])
        overlap[f"{left}_vs_{right}"] = {
            "count": len(values),
            "examples": values[:10],
        }
    missing_major_by_split = {}
    for split, frame in frames.items():
        present = set(str(label) for label in frame.get("major_label", pd.Series(dtype=str)).dropna().unique())
        missing_major_by_split[split] = [label for label in all_major_labels if label not in present]

    warnings = []
    for pair, item in overlap.items():
        if item["count"] > 0:
            warnings.append(f"source_file overlap detected in {pair}: {item['count']}")
    for split, missing in missing_major_by_split.items():
        if missing:
            warnings.append(f"{split} is missing major labels: {', '.join(missing)}")

    return {
        "splits": {split: _split_summary(frame) for split, frame in frames.items()},
        "source_file_overlap": overlap,
        "missing_major_by_split": missing_major_by_split,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed train/val/test parquet splits.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    result = audit(args.train_path, args.val_path, args.test_path)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
