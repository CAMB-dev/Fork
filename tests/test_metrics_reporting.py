import json
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_bert.metrics import attack_detection_metrics, major_classification_metrics
from traffic_bert.metrics_io import export_major_metrics
from traffic_bert.training import write_history_files


def test_major_metrics_include_weighted_and_balanced_scores() -> None:
    metrics = major_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        labels=["benign", "dos_ddos"],
    )

    assert "balanced_accuracy" in metrics
    assert "weighted_precision" in metrics
    assert "weighted_recall" in metrics
    assert metrics["per_class"]["benign"]["support"] == 2.0


def test_attack_detection_metrics_collapses_non_benign_classes() -> None:
    metrics = attack_detection_metrics(
        y_true=[0, 1, 2, 0],
        y_pred=[0, 1, 0, 2],
        labels=["benign", "dos_ddos", "scan"],
    )

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["attack_precision"] == 0.5
    assert metrics["attack_recall"] == 0.5


def test_export_major_metrics_writes_reports_and_normalized_confusion(tmp_path: Path) -> None:
    labels = ["benign", "dos_ddos"]
    metrics = major_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        labels=labels,
    )

    export_major_metrics(tmp_path, metrics, [0, 0, 1, 1], [0, 1, 1, 1], labels)

    report = pd.read_csv(tmp_path / "classification_report.csv")
    assert {"benign", "dos_ddos", "macro avg", "weighted avg"} <= set(report["label"])
    normalized = pd.read_csv(tmp_path / "confusion_matrix_normalized.csv", index_col=0)
    assert np.isclose(normalized.loc["benign"].sum(), 1.0)
    assert np.isclose(normalized.loc["dos_ddos"].sum(), 1.0)
    detection = json.loads((tmp_path / "detection_metrics.json").read_text(encoding="utf-8"))
    assert detection["tp"] == 2


def test_write_history_files_outputs_jsonl_and_csv(tmp_path: Path) -> None:
    history = [
        {
            "epoch": 1,
            "loss": 0.5,
            "learning_rate": 1e-3,
            "val_accuracy": 0.75,
            "val_macro_f1": 0.7,
        }
    ]

    write_history_files(tmp_path, history)

    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8").strip()
    frame = pd.read_csv(tmp_path / "train.csv")
    assert list(frame["epoch"]) == [1]
    assert "val_macro_f1" in frame.columns
