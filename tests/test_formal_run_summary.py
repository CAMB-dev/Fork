from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_formal_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_formal_run", SCRIPT_PATH)
assert SPEC is not None
summary_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary_module)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_base_run(root: Path, *, bot_recall: float, bot_f1: float) -> None:
    _write_json(root / "formal_dataset_gate.json", {"formal_eligible": True})
    _write_json(root / "split_audit.json", {"formal_eligible": True, "blocking_warnings": []})
    _write_json(
        root / "classifier" / "history" / "history.json",
        [
            {
                "epoch": 1,
                "train_accuracy": 0.99,
                "val_accuracy": 0.98,
                "val_macro_f1": 0.85,
            }
        ],
    )
    per_class = {
        "botnet_malware": {
            "precision": 0.90,
            "recall": bot_recall,
            "f1-score": bot_f1,
            "support": 194.0,
        },
        "bruteforce": {"precision": 0.95, "recall": 0.90, "f1-score": 0.92, "support": 1000.0},
        "dos_ddos": {"precision": 0.99, "recall": 0.98, "f1-score": 0.98, "support": 1000.0},
        "scan": {"precision": 0.99, "recall": 0.99, "f1-score": 0.99, "support": 1000.0},
        "web_attack": {"precision": 0.85, "recall": 0.80, "f1-score": 0.82, "support": 200.0},
        "infiltration": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 2.0},
    }
    _write_json(
        root / "classifier" / "test_eval" / "metrics.json",
        {
            "accuracy": 0.99,
            "macro_f1": 0.80,
            "weighted_f1": 0.99,
            "eval_loss": 0.05,
            "per_class": per_class,
        },
    )


def _write_split_with_truncation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "major_label": ["botnet_malware"] * 4 + ["benign"] * 2,
            "was_packet_truncated": [True, True, False, False, False, False],
            "observed_packet_count": [32, 40, 8, 9, 4, 5],
            "packet_count": [16, 16, 8, 9, 4, 5],
        }
    ).to_parquet(path, index=False)


def test_formal_summary_rejects_low_key_class_recall(tmp_path: Path) -> None:
    _write_base_run(tmp_path, bot_recall=0.59, bot_f1=0.68)

    summary = summary_module.summarize(tmp_path)

    acceptance = summary["acceptance"]
    assert acceptance["formal_eligible"] is False
    assert any("botnet_malware recall" in item for item in acceptance["failures"])
    assert acceptance["critical_labels"]["botnet_malware"]["passed"] is False


def test_formal_summary_keeps_host_window_auxiliary_separate(tmp_path: Path) -> None:
    _write_base_run(tmp_path, bot_recall=0.59, bot_f1=0.68)
    _write_json(
        tmp_path / "bot_host_window_analysis.json",
        {
            "target_label": "botnet_malware",
            "flow_probability_threshold": {
                "test_at_chosen": {"precision": 0.80, "recall": 0.60, "f1": 0.69}
            },
            "bucket_window_detection": {
                "30": {
                    "val_at_chosen": {"precision": 0.90, "recall": 0.88, "f1": 0.89},
                    "test_at_chosen": {"precision": 0.87, "recall": 0.84, "f1": 0.85},
                }
            },
        },
    )

    summary = summary_module.summarize(tmp_path)

    assert summary["acceptance"]["formal_eligible"] is False
    auxiliary = summary["auxiliary_host_window"]
    assert auxiliary["available"] is True
    assert auxiliary["auxiliary_eligible"] is True
    assert auxiliary["passed_windows"] == ["30"]
    assert "does not override flow-level acceptance" in auxiliary["scope"]


def test_formal_summary_accepts_key_classes_and_reports_low_support(tmp_path: Path) -> None:
    _write_base_run(tmp_path, bot_recall=0.91, bot_f1=0.88)

    summary = summary_module.summarize(tmp_path)

    acceptance = summary["acceptance"]
    assert acceptance["formal_eligible"] is True
    assert acceptance["failures"] == []
    assert acceptance["critical_labels"]["botnet_malware"]["passed"] is True
    assert any("infiltration" in item for item in acceptance["warnings"])


def test_formal_summary_reports_packet_truncation_warnings(tmp_path: Path) -> None:
    _write_base_run(tmp_path, bot_recall=0.91, bot_f1=0.88)
    split_dir = tmp_path / "split"
    for split in ["train", "val", "test"]:
        _write_split_with_truncation(split_dir / f"{split}.parquet")
    _write_json(
        tmp_path / "formal_dataset_gate.json",
        {
            "formal_eligible": True,
            "checked": {
                "train_path": str(split_dir / "train.parquet"),
                "val_path": str(split_dir / "val.parquet"),
                "test_path": str(split_dir / "test.parquet"),
            },
        },
    )

    summary = summary_module.summarize(
        tmp_path,
        critical_label_overrides=["botnet_malware:1:0.80:0.75"],
        truncation_warn_ratio=0.25,
    )

    assert summary["packet_truncation"]["available"] is True
    assert summary["packet_truncation"]["splits"]["train"]["by_major_label"][
        "botnet_malware"
    ]["truncated_row_ratio"] == 0.5
    assert any("packet truncation ratio" in item for item in summary["acceptance"]["warnings"])
