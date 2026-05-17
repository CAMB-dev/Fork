from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_bert.data.split_audit import audit_split_paths


def audit(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    nearby_threshold_seconds: float = 120.0,
    low_support_min: int = 2,
    low_support_labels: tuple[str, ...] | None = None,
    ignore_source_file_overlap_for_eligibility: bool = False,
    submode_min_train_support: int = 100,
    submode_min_eval_support: int = 100,
    submode_max_unseen_ratio: float = 0.20,
    submode_min_ambiguous_unseen_rows: int = 50,
) -> dict:
    kwargs = {}
    if low_support_labels is not None:
        kwargs["low_support_labels"] = low_support_labels
    return audit_split_paths(
        train_path,
        val_path,
        test_path,
        nearby_threshold_seconds=nearby_threshold_seconds,
        low_support_min=low_support_min,
        ignore_source_file_overlap_for_eligibility=ignore_source_file_overlap_for_eligibility,
        submode_min_train_support=submode_min_train_support,
        submode_min_eval_support=submode_min_eval_support,
        submode_max_unseen_ratio=submode_max_unseen_ratio,
        submode_min_ambiguous_unseen_rows=submode_min_ambiguous_unseen_rows,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed train/val/test parquet splits.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--nearby-threshold-seconds", type=float, default=120.0)
    parser.add_argument("--low-support-min", type=int, default=2)
    parser.add_argument(
        "--low-support-label",
        action="append",
        default=None,
        help="Major label that must have at least --low-support-min rows in val/test. "
        "May be provided multiple times.",
    )
    parser.add_argument("--ignore-source-file-overlap-for-eligibility", action="store_true")
    parser.add_argument("--submode-min-train-support", type=int, default=100)
    parser.add_argument("--submode-min-eval-support", type=int, default=100)
    parser.add_argument("--submode-max-unseen-ratio", type=float, default=0.20)
    parser.add_argument("--submode-min-ambiguous-unseen-rows", type=int, default=50)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    result = audit(
        args.train_path,
        args.val_path,
        args.test_path,
        nearby_threshold_seconds=args.nearby_threshold_seconds,
        low_support_min=args.low_support_min,
        low_support_labels=None
        if args.low_support_label is None
        else tuple(args.low_support_label),
        ignore_source_file_overlap_for_eligibility=args.ignore_source_file_overlap_for_eligibility,
        submode_min_train_support=args.submode_min_train_support,
        submode_min_eval_support=args.submode_min_eval_support,
        submode_max_unseen_ratio=args.submode_max_unseen_ratio,
        submode_min_ambiguous_unseen_rows=args.submode_min_ambiguous_unseen_rows,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_warnings and not result.get("formal_eligible", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
