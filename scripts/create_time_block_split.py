"""Create a stratified shuffled time-block train/val/test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_bert.config import write_json
from traffic_bert.data.split import (
    assign_time_block_split,
    split_run_stats,
    stratified_sample,
)
from traffic_bert.data.validate import validation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stratify-column", default="major_label")
    parser.add_argument("--split-stratify-column", default="source_label")
    parser.add_argument("--group-column", default="flow_id")
    parser.add_argument("--time-column", default="start_time")
    parser.add_argument("--max-per-class", type=int, default=50_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input_path)
    input_frame = frame
    if args.max_per_class > 0:
        frame = stratified_sample(
            frame,
            stratify_column=args.stratify_column,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )
    frame = assign_time_block_split(
        frame,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        group_column=args.group_column,
        stratify_column=args.split_stratify_column,
        time_column=args.time_column,
        block_size=args.block_size,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frame = frame[frame["split"] == split_name]
        split_frame.to_parquet(args.output_dir / f"{split_name}.parquet", index=False)
        split_frame["major_label"].value_counts().rename_axis("major_label").reset_index(
            name="rows"
        ).to_csv(args.output_dir / f"{split_name}.class_distribution.csv", index=False)
        write_json(
            args.output_dir / f"{split_name}.validate.json",
            validation_summary(split_frame),
        )

    stats = split_run_stats(
        input_frame=input_frame,
        output_frame=frame,
        split_method="time_block",
        max_per_class=args.max_per_class,
        stratify_column=args.stratify_column,
        split_stratify_column=args.split_stratify_column,
        group_column=args.group_column,
        time_column=args.time_column,
        block_size=args.block_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_json(args.output_dir / "split.stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
