"""Summarize formal CICIDS training artifacts and basic overfit signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_CRITICAL_LABELS = {
    "botnet_malware": {"min_support": 100, "min_recall": 0.80, "min_f1": 0.75},
    "bruteforce": {"min_support": 100, "min_recall": 0.80, "min_f1": 0.75},
    "dos_ddos": {"min_support": 100, "min_recall": 0.90, "min_f1": 0.90},
    "scan": {"min_support": 100, "min_recall": 0.90, "min_f1": 0.90},
    "web_attack": {"min_support": 100, "min_recall": 0.70, "min_f1": 0.70},
}
LOW_SUPPORT_REPORTED_ONLY = {"heartbleed", "infiltration"}


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _status_from_gate(payload: dict | None) -> str:
    if payload is None:
        return "missing"
    if payload.get("formal_eligible") is True:
        return "pass"
    return "fail"


def _last(history: list[dict[str, Any]]) -> dict[str, Any]:
    return history[-1] if history else {}


def _parse_critical_label(value: str) -> tuple[str, dict[str, float | int]]:
    parts = [item.strip() for item in value.split(":")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "critical labels must use label:min_support:min_recall:min_f1"
        )
    label, min_support, min_recall, min_f1 = parts
    if not label:
        raise argparse.ArgumentTypeError("critical label name cannot be empty")
    return (
        label,
        {
            "min_support": int(min_support),
            "min_recall": float(min_recall),
            "min_f1": float(min_f1),
        },
    )


def _critical_label_config(overrides: list[str] | None = None) -> dict[str, dict[str, float | int]]:
    config = {label: dict(values) for label, values in DEFAULT_CRITICAL_LABELS.items()}
    for raw in overrides or []:
        label, values = _parse_critical_label(raw)
        config[label] = values
    return config


def _metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _acceptance_report(
    *,
    gate: dict | None,
    split_audit: dict | None,
    history: list[dict[str, Any]],
    test_metrics: dict | None,
    critical_labels: dict[str, dict[str, float | int]],
    truncation: dict[str, Any] | None = None,
    truncation_warn_ratio: float = 0.25,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []

    if _status_from_gate(gate) != "pass":
        failures.append("formal dataset gate did not pass")
    if _status_from_gate(split_audit) != "pass":
        failures.append("split audit did not pass")
    if not history:
        failures.append("classifier training history is missing")
    if test_metrics is None:
        failures.append("classifier test metrics are missing")

    test_report = {} if test_metrics is None else test_metrics.get("per_class", {})
    critical_results: dict[str, dict[str, float | int | bool]] = {}
    for label, thresholds in critical_labels.items():
        row = test_report.get(label, {})
        support = _metric_value(row, "support")
        recall = _metric_value(row, "recall")
        f1 = _metric_value(row, "f1-score")
        min_support = int(thresholds["min_support"])
        min_recall = float(thresholds["min_recall"])
        min_f1 = float(thresholds["min_f1"])
        passed = support >= min_support and recall >= min_recall and f1 >= min_f1
        critical_results[label] = {
            "support": support,
            "recall": recall,
            "f1": f1,
            "min_support": min_support,
            "min_recall": min_recall,
            "min_f1": min_f1,
            "passed": passed,
        }
        if test_metrics is None:
            continue
        if support < min_support:
            failures.append(
                f"{label} support {support:g} < required {min_support}; cannot validate key class"
            )
            continue
        if recall < min_recall:
            failures.append(f"{label} recall {recall:.4f} < required {min_recall:.4f}")
        if f1 < min_f1:
            failures.append(f"{label} f1 {f1:.4f} < required {min_f1:.4f}")

    for label in sorted(LOW_SUPPORT_REPORTED_ONLY):
        row = test_report.get(label, {})
        support = _metric_value(row, "support")
        if support > 0:
            warnings.append(
                f"{label} has test support {support:g}; report it as low-support only"
            )

    for split_name, split_report in (truncation or {}).get("splits", {}).items():
        for label in critical_labels:
            label_report = split_report.get("by_major_label", {}).get(label)
            if not label_report:
                continue
            rows = int(label_report.get("rows", 0))
            ratio = float(label_report.get("truncated_row_ratio", 0.0))
            min_support = int(critical_labels[label]["min_support"])
            if rows >= min_support and ratio > truncation_warn_ratio:
                warnings.append(
                    f"{split_name} {label} packet truncation ratio "
                    f"{ratio:.3f} > warning threshold {truncation_warn_ratio:.3f}"
                )

    if history:
        best = max(history, key=lambda row: float(row.get("val_macro_f1", float("-inf"))))
        last = _last(history)
        best_f1 = best.get("val_macro_f1")
        last_f1 = last.get("val_macro_f1")
        if best_f1 is not None and last_f1 is not None and float(best_f1) - float(last_f1) > 0.05:
            warnings.append(
                "last epoch val_macro_f1 is more than 0.05 below best epoch; "
                "use classifier.best.pt for reporting"
            )

    return {
        "formal_eligible": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "critical_labels": critical_results,
    }


def _gate_split_paths(gate: dict | None) -> dict[str, Path]:
    checked = {} if gate is None else gate.get("checked", {})
    paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        raw = checked.get(f"{split_name}_path")
        if raw:
            paths[split_name] = Path(raw)
    return paths


def _truncation_report(gate: dict | None) -> dict[str, Any]:
    split_paths = _gate_split_paths(gate)
    report: dict[str, Any] = {
        "available": False,
        "splits": {},
    }
    required_columns = ["major_label", "was_packet_truncated"]
    optional_columns = ["observed_packet_count", "packet_count"]
    for split_name, path in split_paths.items():
        if not path.exists():
            continue
        schema = set(pq.read_schema(path).names)
        columns = list(required_columns)
        for column in optional_columns:
            if column in schema:
                columns.append(column)
        frame = pd.read_parquet(path, columns=columns)
        if not set(required_columns) <= set(frame.columns):
            continue
        truncated = frame["was_packet_truncated"].fillna(False).astype(bool)
        split_payload: dict[str, Any] = {
            "rows": int(len(frame)),
            "truncated_rows": int(truncated.sum()),
            "truncated_row_ratio": float(truncated.mean()) if len(frame) else 0.0,
            "by_major_label": {},
        }
        for label, group in frame.assign(_truncated=truncated).groupby("major_label"):
            group_truncated = group["_truncated"].astype(bool)
            label_payload: dict[str, Any] = {
                "rows": int(len(group)),
                "truncated_rows": int(group_truncated.sum()),
                "truncated_row_ratio": float(group_truncated.mean()) if len(group) else 0.0,
            }
            if "observed_packet_count" in group.columns:
                label_payload["observed_packet_count_max"] = int(
                    group["observed_packet_count"].max()
                )
            if "packet_count" in group.columns:
                label_payload["packet_count_max"] = int(group["packet_count"].max())
            split_payload["by_major_label"][str(label)] = label_payload
        report["splits"][split_name] = split_payload
    report["available"] = bool(report["splits"])
    return report


def _host_window_auxiliary_report(
    path: Path | None,
    *,
    min_recall: float,
    min_f1: float,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "available": False,
            "auxiliary_eligible": False,
            "path": None if path is None else str(path),
            "scope": "host-window risk layer only; does not override flow-level acceptance",
        }
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {
            "available": False,
            "auxiliary_eligible": False,
            "path": str(path),
            "error": "host-window analysis is not a JSON object",
            "scope": "host-window risk layer only; does not override flow-level acceptance",
        }

    bucket_windows: dict[str, Any] = {}
    passed_windows: list[str] = []
    for seconds, window_report in sorted(
        payload.get("bucket_window_detection", {}).items(),
        key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])),
    ):
        val_metrics = dict(window_report.get("val_at_chosen", {}))
        test_metrics = dict(window_report.get("test_at_chosen", {}))
        val_passed = (
            float(val_metrics.get("recall", 0.0)) >= min_recall
            and float(val_metrics.get("f1", 0.0)) >= min_f1
        )
        test_passed = (
            float(test_metrics.get("recall", 0.0)) >= min_recall
            and float(test_metrics.get("f1", 0.0)) >= min_f1
        )
        if val_passed and test_passed:
            passed_windows.append(str(seconds))
        bucket_windows[str(seconds)] = {
            "val_at_chosen": val_metrics,
            "test_at_chosen": test_metrics,
            "val_passed_requirements": val_passed,
            "test_passed_requirements": test_passed,
            "passed_requirements": val_passed and test_passed,
        }

    return {
        "available": True,
        "auxiliary_eligible": bool(passed_windows),
        "path": str(path),
        "target_label": payload.get("target_label"),
        "min_recall": min_recall,
        "min_f1": min_f1,
        "passed_windows": passed_windows,
        "bucket_windows": bucket_windows,
        "flow_probability_test_at_chosen": payload.get("flow_probability_threshold", {}).get(
            "test_at_chosen"
        ),
        "scope": "host-window risk layer only; does not override flow-level acceptance",
    }


def summarize(
    output_root: Path,
    *,
    critical_label_overrides: list[str] | None = None,
    truncation_warn_ratio: float = 0.25,
    host_window_analysis_path: Path | None = None,
    host_window_min_recall: float = 0.80,
    host_window_min_f1: float = 0.75,
) -> dict[str, Any]:
    classifier_dir = output_root / "classifier"
    history = _read_json(classifier_dir / "history" / "history.json") or []
    test_metrics = _read_json(classifier_dir / "test_eval" / "metrics.json")
    gate = _read_json(output_root / "formal_dataset_gate.json")
    split_audit = _read_json(output_root / "split_audit.json")
    truncation = _truncation_report(gate)
    if host_window_analysis_path is None:
        host_window_analysis_path = output_root / "bot_host_window_analysis.json"
    auxiliary_host_window = _host_window_auxiliary_report(
        host_window_analysis_path,
        min_recall=host_window_min_recall,
        min_f1=host_window_min_f1,
    )

    last_epoch = _last(history)
    best_epoch = None
    if history:
        best_epoch = max(
            history,
            key=lambda row: float(row.get("val_macro_f1", float("-inf"))),
        )

    train_accuracy = last_epoch.get("train_accuracy")
    val_accuracy = last_epoch.get("val_accuracy")
    accuracy_gap = None
    if train_accuracy is not None and val_accuracy is not None:
        accuracy_gap = float(train_accuracy) - float(val_accuracy)

    critical_labels = _critical_label_config(critical_label_overrides)
    acceptance = _acceptance_report(
        gate=gate,
        split_audit=split_audit,
        history=history,
        test_metrics=test_metrics,
        critical_labels=critical_labels,
        truncation=truncation,
        truncation_warn_ratio=truncation_warn_ratio,
    )

    return {
        "output_root": str(output_root),
        "formal_gate": _status_from_gate(gate),
        "split_audit": _status_from_gate(split_audit),
        "history_exists": bool(history),
        "epochs": len(history),
        "last_epoch": {
            "epoch": last_epoch.get("epoch"),
            "train_loss": last_epoch.get("train_loss"),
            "train_accuracy": train_accuracy,
            "val_loss": last_epoch.get("val_loss"),
            "val_accuracy": val_accuracy,
            "val_macro_f1": last_epoch.get("val_macro_f1"),
            "val_weighted_f1": last_epoch.get("val_weighted_f1"),
        },
        "best_val_epoch": None
        if best_epoch is None
        else {
            "epoch": best_epoch.get("epoch"),
            "val_macro_f1": best_epoch.get("val_macro_f1"),
            "val_accuracy": best_epoch.get("val_accuracy"),
            "val_loss": best_epoch.get("val_loss"),
        },
        "test_metrics_exists": test_metrics is not None,
        "test_metrics": {}
        if test_metrics is None
        else {
            "accuracy": test_metrics.get("accuracy"),
            "macro_f1": test_metrics.get("macro_f1"),
            "weighted_f1": test_metrics.get("weighted_f1"),
            "eval_loss": test_metrics.get("eval_loss"),
        },
        "overfit_signals": {
            "train_val_accuracy_gap": accuracy_gap,
            "warning": accuracy_gap is not None and accuracy_gap > 0.15,
        },
        "packet_truncation": truncation,
        "auxiliary_host_window": auxiliary_host_window,
        "acceptance": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument(
        "--critical-label",
        action="append",
        default=[],
        help="Override/add a key-class gate as label:min_support:min_recall:min_f1.",
    )
    parser.add_argument(
        "--fail-on-acceptance",
        action="store_true",
        help="Exit non-zero when formal run acceptance fails.",
    )
    parser.add_argument(
        "--warn-truncation-ratio",
        type=float,
        default=0.25,
        help="Warn when a critical label exceeds this packet truncation ratio.",
    )
    parser.add_argument(
        "--host-window-analysis-path",
        type=Path,
        default=None,
        help="Optional bot/host-window analysis JSON to include as an auxiliary risk layer.",
    )
    parser.add_argument(
        "--host-window-min-recall",
        type=float,
        default=0.80,
        help="Minimum recall for auxiliary host-window detection.",
    )
    parser.add_argument(
        "--host-window-min-f1",
        type=float,
        default=0.75,
        help="Minimum F1 for auxiliary host-window detection.",
    )
    args = parser.parse_args()

    summary = summarize(
        args.output_root,
        critical_label_overrides=args.critical_label,
        truncation_warn_ratio=args.warn_truncation_ratio,
        host_window_analysis_path=args.host_window_analysis_path,
        host_window_min_recall=args.host_window_min_recall,
        host_window_min_f1=args.host_window_min_f1,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(payload + "\n", encoding="utf-8")
    if args.fail_on_acceptance and not summary["acceptance"]["formal_eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
