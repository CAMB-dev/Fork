from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_bert.data.split_audit import audit_split_paths


def audit(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    nearby_threshold_seconds: float = 60.0,
    low_support_min: int = 2,
) -> dict:
    return audit_split_paths(
        train_path,
        val_path,
        test_path,
        nearby_threshold_seconds=nearby_threshold_seconds,
        low_support_min=low_support_min,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed train/val/test parquet splits.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--nearby-threshold-seconds", type=float, default=60.0)
    parser.add_argument("--low-support-min", type=int, default=2)
    args = parser.parse_args()

    result = audit(
        args.train_path,
        args.val_path,
        args.test_path,
        nearby_threshold_seconds=args.nearby_threshold_seconds,
        low_support_min=args.low_support_min,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
