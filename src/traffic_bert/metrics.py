"""Evaluation metrics for major and minor labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score


def major_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    label_ids = list(range(len(labels)))
    present_ids = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "present_class_macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "present_class_macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "present_class_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "all_config_macro_precision": float(per_class["macro avg"]["precision"]),
        "all_config_macro_recall": float(per_class["macro avg"]["recall"]),
        "all_config_macro_f1": float(per_class["macro avg"]["f1-score"]),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "present_label_ids": present_ids,
        "per_class": per_class,
    }


def attack_detection_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str],
    benign_label: str = "benign",
) -> dict[str, float | int]:
    """Collapse major labels into benign-vs-attack detection metrics."""

    benign_id = labels.index(benign_label)
    y_true_attack = np.asarray(y_true) != benign_id
    y_pred_attack = np.asarray(y_pred) != benign_id

    tp = int(np.logical_and(y_true_attack, y_pred_attack).sum())
    tn = int(np.logical_and(~y_true_attack, ~y_pred_attack).sum())
    fp = int(np.logical_and(~y_true_attack, y_pred_attack).sum())
    fn = int(np.logical_and(y_true_attack, ~y_pred_attack).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "attack_precision": float(precision),
        "attack_recall": float(recall),
        "attack_tpr": float(recall),
        "attack_f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "binary_accuracy": float(accuracy),
    }


def multilabel_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= thresholds).astype(int)
    return {
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def constrain_minor_predictions(
    y_pred: np.ndarray,
    major_ids: np.ndarray,
    minor_to_major_ids: list[int],
) -> np.ndarray:
    constrained = np.asarray(y_pred).copy()
    mapping = np.asarray(minor_to_major_ids)
    if constrained.size == 0 or mapping.size == 0:
        return constrained
    allowed = mapping.reshape(1, -1) == np.asarray(major_ids).reshape(-1, 1)
    constrained[~allowed] = 0
    return constrained


def multilabel_scores_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def multilabel_report_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )


def multilabel_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= thresholds).astype(int)
    return classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
