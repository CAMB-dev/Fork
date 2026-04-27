"""Evaluation metrics for major and minor labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
from sklearn.metrics import recall_score


def major_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    label_ids = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def multilabel_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= thresholds).astype(int)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
