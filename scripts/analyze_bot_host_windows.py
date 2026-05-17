from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from traffic_bert.config import write_json


BASE_SPLIT_COLUMNS = [
    "flow_id",
    "major_label",
    "source_file",
    "endpoint_a",
    "endpoint_b",
    "start_time",
    "connection_type",
    "payload_byte_length",
]
BASE_PREDICTION_COLUMNS = ["flow_id", "true_major_label", "pred_major_label"]


def _host(endpoint: Any) -> str:
    text = "" if endpoint is None else str(endpoint)
    if ":" not in text:
        return text
    return text.rsplit(":", 1)[0]


def _available_columns(path: Path, requested: list[str]) -> list[str]:
    names = set(pq.read_schema(path).names)
    return [column for column in requested if column in names]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
    }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.asarray([0.5], dtype=float)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 301))
    grid = np.linspace(0.001, 0.999, 999)
    return np.unique(np.concatenate([grid, quantiles]))


def scan_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float,
    min_f1: float,
) -> dict[str, Any]:
    candidates = []
    for threshold in _candidate_thresholds(scores):
        metrics = binary_metrics(y_true, np.asarray(scores) >= float(threshold))
        candidates.append({"threshold": float(threshold), **metrics})
    passing = [
        item
        for item in candidates
        if item["recall"] >= min_recall and item["f1"] >= min_f1
    ]
    pool = passing or candidates
    chosen = max(
        pool,
        key=lambda item: (
            float(item["f1"]),
            float(item["recall"]),
            float(item["precision"]),
            -int(item["fp"]),
            float(item["threshold"]),
        ),
    )
    return {
        "chosen": chosen,
        "chosen_passed_requirements": bool(passing),
        "top_candidates": sorted(
            candidates,
            key=lambda item: (float(item["f1"]), float(item["recall"]), float(item["precision"])),
            reverse=True,
        )[:10],
    }


def apply_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {"threshold": float(threshold), **binary_metrics(y_true, np.asarray(scores) >= threshold)}


def load_joined_split(
    *,
    split_path: Path,
    prediction_path: Path,
    score_column: str,
    target_label: str,
) -> pd.DataFrame:
    split_columns = _available_columns(split_path, BASE_SPLIT_COLUMNS)
    missing_split = {"flow_id", "major_label", "endpoint_a", "start_time"} - set(split_columns)
    if missing_split:
        raise KeyError(f"{split_path} missing required columns: {sorted(missing_split)}")
    prediction_columns = _available_columns(
        prediction_path,
        [*BASE_PREDICTION_COLUMNS, score_column],
    )
    missing_predictions = {"flow_id", "pred_major_label", score_column} - set(prediction_columns)
    if missing_predictions:
        raise KeyError(
            f"{prediction_path} missing required columns: {sorted(missing_predictions)}"
        )
    split_frame = pd.read_parquet(split_path, columns=split_columns)
    prediction_frame = pd.read_parquet(prediction_path, columns=prediction_columns)
    joined = split_frame.merge(
        prediction_frame,
        on="flow_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(split_frame) or len(joined) != len(prediction_frame):
        raise ValueError(
            "split/prediction row mismatch after joining on flow_id: "
            f"split={len(split_frame)} predictions={len(prediction_frame)} joined={len(joined)}"
        )
    joined["_is_target"] = joined["major_label"].astype(str).eq(target_label)
    joined["_pred_is_target"] = joined["pred_major_label"].astype(str).eq(target_label)
    joined["_score"] = joined[score_column].map(_safe_float)
    joined["_source_host"] = joined["endpoint_a"].map(_host)
    if "source_file" not in joined.columns:
        joined["source_file"] = ""
    return joined


def add_trailing_window_scores(frame: pd.DataFrame, window_seconds: int) -> pd.Series:
    scores = pd.Series(0.0, index=frame.index, dtype="float64")
    group_columns = ["source_file", "_source_host"]
    for _, group in frame.groupby(group_columns, sort=False):
        items = list(group[["start_time", "_score"]].itertuples(index=True, name=None))
        items.sort(key=lambda item: (float(item[1]), int(item[0])))
        window: deque[tuple[int, float, float]] = deque()
        for index, start_time, score in items:
            start_time = float(start_time)
            score = float(score)
            window.append((int(index), start_time, score))
            while window and start_time - float(window[0][1]) > float(window_seconds):
                window.popleft()
            scores.at[int(index)] = max(item[2] for item in window)
    return scores


def bucketize_windows(frame: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    work = frame[["source_file", "_source_host", "start_time", "_is_target", "_score"]].copy()
    work["_bucket"] = np.floor(work["start_time"].astype(float) / float(window_seconds)).astype(
        "int64"
    )
    grouped = (
        work.groupby(["source_file", "_source_host", "_bucket"], sort=False)
        .agg(
            is_target=("_is_target", "max"),
            score=("_score", "max"),
            rows=("_score", "size"),
            target_rows=("_is_target", "sum"),
        )
        .reset_index()
    )
    grouped["is_target"] = grouped["is_target"].astype(bool)
    return grouped


def split_overview(frame: pd.DataFrame, target_label: str) -> dict[str, Any]:
    overview: dict[str, Any] = {
        "rows": int(len(frame)),
        "target_label": target_label,
        "target_support": int(frame["_is_target"].sum()),
        "source_host_count": int(frame["_source_host"].nunique()),
        "target_source_host_count": int(frame.loc[frame["_is_target"], "_source_host"].nunique()),
        "argmax": binary_metrics(frame["_is_target"].to_numpy(), frame["_pred_is_target"].to_numpy()),
    }
    if "connection_type" in frame.columns:
        target_connections = frame.loc[frame["_is_target"], "connection_type"].astype(str)
        overview["target_connection_type_counts"] = {
            str(key): int(value) for key, value in target_connections.value_counts().to_dict().items()
        }
    if "payload_byte_length" in frame.columns:
        target_payload = frame.loc[frame["_is_target"], "payload_byte_length"].fillna(0)
        overview["target_payload_byte_length_quantiles"] = {
            str(q): float(target_payload.quantile(q)) if len(target_payload) else 0.0
            for q in [0.0, 0.5, 0.9, 0.99, 1.0]
        }
    return overview


def analyze(
    *,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_label: str,
    window_seconds: list[int],
    min_recall: float,
    min_f1: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target_label": target_label,
        "min_recall": min_recall,
        "min_f1": min_f1,
        "splits": {
            "val": split_overview(val_frame, target_label),
            "test": split_overview(test_frame, target_label),
        },
    }

    flow_scan = scan_threshold(
        val_frame["_is_target"].to_numpy(),
        val_frame["_score"].to_numpy(),
        min_recall=min_recall,
        min_f1=min_f1,
    )
    flow_threshold = float(flow_scan["chosen"]["threshold"])
    result["flow_probability_threshold"] = {
        "val_scan": flow_scan,
        "val_at_chosen": apply_threshold(
            val_frame["_is_target"].to_numpy(),
            val_frame["_score"].to_numpy(),
            flow_threshold,
        ),
        "test_at_chosen": apply_threshold(
            test_frame["_is_target"].to_numpy(),
            test_frame["_score"].to_numpy(),
            flow_threshold,
        ),
    }

    trailing: dict[str, Any] = {}
    buckets: dict[str, Any] = {}
    cooccurrence: dict[str, Any] = {}
    for seconds in window_seconds:
        key = str(int(seconds))
        val_trailing = add_trailing_window_scores(val_frame, int(seconds))
        test_trailing = add_trailing_window_scores(test_frame, int(seconds))
        scan = scan_threshold(
            val_frame["_is_target"].to_numpy(),
            val_trailing.to_numpy(),
            min_recall=min_recall,
            min_f1=min_f1,
        )
        threshold = float(scan["chosen"]["threshold"])
        trailing[key] = {
            "val_scan": scan,
            "val_at_chosen": apply_threshold(
                val_frame["_is_target"].to_numpy(), val_trailing.to_numpy(), threshold
            ),
            "test_at_chosen": apply_threshold(
                test_frame["_is_target"].to_numpy(), test_trailing.to_numpy(), threshold
            ),
        }

        val_buckets = bucketize_windows(val_frame, int(seconds))
        test_buckets = bucketize_windows(test_frame, int(seconds))
        bucket_scan = scan_threshold(
            val_buckets["is_target"].to_numpy(),
            val_buckets["score"].to_numpy(),
            min_recall=min_recall,
            min_f1=min_f1,
        )
        bucket_threshold = float(bucket_scan["chosen"]["threshold"])
        buckets[key] = {
            "val_windows": int(len(val_buckets)),
            "val_positive_windows": int(val_buckets["is_target"].sum()),
            "test_windows": int(len(test_buckets)),
            "test_positive_windows": int(test_buckets["is_target"].sum()),
            "val_scan": bucket_scan,
            "val_at_chosen": apply_threshold(
                val_buckets["is_target"].to_numpy(),
                val_buckets["score"].to_numpy(),
                bucket_threshold,
            ),
            "test_at_chosen": apply_threshold(
                test_buckets["is_target"].to_numpy(),
                test_buckets["score"].to_numpy(),
                bucket_threshold,
            ),
        }

        split_cooccurrence: dict[str, Any] = {}
        for split_name, frame in [("val", val_frame), ("test", test_frame)]:
            buckets_for_split = bucketize_windows(frame, int(seconds))
            positive_keys = buckets_for_split.loc[
                buckets_for_split["is_target"], ["source_file", "_source_host", "_bucket"]
            ]
            work = frame.copy()
            work["_bucket"] = np.floor(work["start_time"].astype(float) / float(seconds)).astype(
                "int64"
            )
            merged = work.merge(
                positive_keys.assign(_target_window=True),
                on=["source_file", "_source_host", "_bucket"],
                how="left",
            )
            in_target_window = merged["_target_window"].fillna(False).astype(bool)
            split_cooccurrence[split_name] = {
                "rows_in_target_windows": int(in_target_window.sum()),
                "target_rows_in_target_windows": int((in_target_window & merged["_is_target"]).sum()),
                "benign_rows_in_target_windows": int(
                    (in_target_window & ~merged["_is_target"]).sum()
                ),
            }
        cooccurrence[key] = split_cooccurrence

    result["trailing_window_flow_projection"] = trailing
    result["bucket_window_detection"] = buckets
    result["target_window_cooccurrence"] = cooccurrence
    result["policies"] = [
        {
            "kind": "host_window_bucket_max_probability",
            "target_label": target_label,
            "window_seconds": int(seconds),
            "score_column": f"prob_{target_label}",
            "threshold": float(report["val_at_chosen"]["threshold"]),
            "selected_on": "val",
            "val_metrics": report["val_at_chosen"],
            "test_metrics": report["test_at_chosen"],
            "val_passed_requirements": bool(
                report["val_at_chosen"]["recall"] >= min_recall
                and report["val_at_chosen"]["f1"] >= min_f1
            ),
            "test_passed_requirements": bool(
                report["test_at_chosen"]["recall"] >= min_recall
                and report["test_at_chosen"]["f1"] >= min_f1
            ),
            "enabled": bool(
                report["val_at_chosen"]["recall"] >= min_recall
                and report["val_at_chosen"]["f1"] >= min_f1
                and report["test_at_chosen"]["recall"] >= min_recall
                and report["test_at_chosen"]["f1"] >= min_f1
            ),
        }
        for seconds, report in sorted(buckets.items(), key=lambda item: int(item[0]))
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether a target label needs host-window behavior context."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-label", default="botnet_malware")
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--window-seconds", type=int, nargs="+", default=[60, 300])
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-f1", type=float, default=0.75)
    args = parser.parse_args()

    score_column = args.score_column or f"prob_{args.target_label}"
    val_frame = load_joined_split(
        split_path=args.split_dir / "val.parquet",
        prediction_path=args.prediction_dir / "val_predictions.parquet",
        score_column=score_column,
        target_label=args.target_label,
    )
    test_frame = load_joined_split(
        split_path=args.split_dir / "test.parquet",
        prediction_path=args.prediction_dir / "test_predictions.parquet",
        score_column=score_column,
        target_label=args.target_label,
    )
    result = analyze(
        val_frame=val_frame,
        test_frame=test_frame,
        target_label=args.target_label,
        window_seconds=args.window_seconds,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
    )
    result["inputs"] = {
        "split_dir": str(args.split_dir),
        "prediction_dir": str(args.prediction_dir),
        "score_column": score_column,
        "window_seconds": [int(item) for item in args.window_seconds],
    }
    write_json(args.output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
