"""Threshold calibration for hierarchical minor-label outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdSearchResult:
    label: str
    threshold: float
    f1: float
    precision: float
    recall: float
    support: int


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    candidate_thresholds: np.ndarray | None = None,
    default_threshold: float = 0.5,
) -> tuple[dict[str, float], list[ThresholdSearchResult]]:
    """Search per-label thresholds that maximize validation F1.

    Labels with no positive validation examples keep ``default_threshold`` to
    avoid fitting a threshold to pure negatives.
    """

    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if y_true.shape[1] != len(labels):
        raise ValueError("number of labels does not match y_true/y_prob width")
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 19)

    thresholds: dict[str, float] = {}
    results: list[ThresholdSearchResult] = []
    for idx, label in enumerate(labels):
        true_col = y_true[:, idx].astype(int)
        prob_col = y_prob[:, idx]
        support = int(true_col.sum())
        if support == 0:
            precision, recall, f1 = _binary_scores(
                true_col,
                (prob_col >= default_threshold).astype(int),
            )
            thresholds[label] = default_threshold
            results.append(
                ThresholdSearchResult(
                    label=label,
                    threshold=default_threshold,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    support=support,
                )
            )
            continue

        best = ThresholdSearchResult(
            label=label,
            threshold=default_threshold,
            f1=-1.0,
            precision=0.0,
            recall=0.0,
            support=support,
        )
        for threshold in candidate_thresholds:
            pred_col = (prob_col >= float(threshold)).astype(int)
            precision, recall, f1 = _binary_scores(true_col, pred_col)
            if f1 > best.f1 or (f1 == best.f1 and float(threshold) > best.threshold):
                best = ThresholdSearchResult(
                    label=label,
                    threshold=float(threshold),
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    support=support,
                )
        thresholds[label] = best.threshold
        results.append(best)

    return thresholds, results

