"""Metric export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix


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
    rows = []
    for label in labels:
        if label in per_class:
            row = {"label": label}
            row.update(per_class[label])
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "per_class_metrics.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv"
    )

