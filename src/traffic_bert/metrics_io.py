"""Metric export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

from traffic_bert.metrics import attack_detection_metrics


def _report_rows(report: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for label in [*labels, "accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"]:
        value = report.get(label)
        if value is None:
            continue
        if isinstance(value, dict):
            row = {"label": label}
            row.update(value)
        else:
            row = {"label": label, "precision": "", "recall": "", "f1-score": value, "support": ""}
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_major_metrics(
    output_dir: str | Path,
    metrics: dict[str, Any],
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_class = metrics.get("per_class", {})
    _write_json(output_dir / "classification_report.json", per_class)
    rows = _report_rows(per_class, labels)
    if rows:
        report_frame = pd.DataFrame(rows)
        report_frame.to_csv(output_dir / "classification_report.csv", index=False)
        report_frame[report_frame["label"].isin(labels)].to_csv(
            output_dir / "per_class_metrics.csv",
            index=False,
        )

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix.astype(float)
    normalized = normalized / row_sums.clip(1)
    pd.DataFrame(normalized, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )

    detection = attack_detection_metrics(y_true, y_pred, labels)
    _write_json(output_dir / "detection_metrics.json", detection)


def export_minor_metrics(
    output_dir: str | Path,
    report: dict[str, Any],
    labels: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "minor_classification_report.json", report)
    rows = _report_rows(report, labels)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "minor_classification_report.csv", index=False)
