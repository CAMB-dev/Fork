from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_bot_host_windows.py"
SPEC = importlib.util.spec_from_file_location("analyze_bot_host_windows", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        _is_target=lambda frame: frame["major_label"].eq("botnet_malware"),
        _pred_is_target=lambda frame: frame["pred_major_label"].eq("botnet_malware"),
        _score=lambda frame: frame["prob_botnet_malware"].astype(float),
        _source_host=lambda frame: frame["endpoint_a"].map(MODULE._host),
    )


def test_trailing_window_score_uses_prior_same_host_only() -> None:
    frame = _frame(
        [
            {
                "flow_id": "a",
                "major_label": "benign",
                "pred_major_label": "botnet_malware",
                "prob_botnet_malware": 0.90,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.1:1111",
                "start_time": 100.0,
            },
            {
                "flow_id": "b",
                "major_label": "botnet_malware",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.20,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.1:2222",
                "start_time": 110.0,
            },
            {
                "flow_id": "c",
                "major_label": "botnet_malware",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.10,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.2:3333",
                "start_time": 115.0,
            },
            {
                "flow_id": "d",
                "major_label": "botnet_malware",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.30,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.1:4444",
                "start_time": 170.5,
            },
        ]
    )

    scores = MODULE.add_trailing_window_scores(frame, 60).tolist()

    assert scores == [0.90, 0.90, 0.10, 0.30]


def test_analyze_reports_window_detection_separately_from_flow_projection() -> None:
    val = _frame(
        [
            {
                "flow_id": "v1",
                "major_label": "benign",
                "pred_major_label": "botnet_malware",
                "prob_botnet_malware": 0.90,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.1:1111",
                "start_time": 100.0,
            },
            {
                "flow_id": "v2",
                "major_label": "botnet_malware",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.30,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.1:2222",
                "start_time": 110.0,
            },
            {
                "flow_id": "v3",
                "major_label": "benign",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.01,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.2:3333",
                "start_time": 110.0,
            },
        ]
    )
    test = _frame(
        [
            {
                "flow_id": "t1",
                "major_label": "benign",
                "pred_major_label": "botnet_malware",
                "prob_botnet_malware": 0.85,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.3:1111",
                "start_time": 200.0,
            },
            {
                "flow_id": "t2",
                "major_label": "botnet_malware",
                "pred_major_label": "benign",
                "prob_botnet_malware": 0.25,
                "source_file": "day.pcap",
                "endpoint_a": "10.0.0.3:2222",
                "start_time": 210.0,
            },
        ]
    )

    result = MODULE.analyze(
        val_frame=val,
        test_frame=test,
        target_label="botnet_malware",
        window_seconds=[60],
        min_recall=0.8,
        min_f1=0.75,
    )

    assert result["splits"]["val"]["argmax"]["recall"] == 0.0
    assert result["bucket_window_detection"]["60"]["val_at_chosen"]["recall"] == 1.0
    assert result["target_window_cooccurrence"]["60"]["val"]["benign_rows_in_target_windows"] == 1
    assert result["policies"] == [
        {
            "kind": "host_window_bucket_max_probability",
            "target_label": "botnet_malware",
            "window_seconds": 60,
            "score_column": "prob_botnet_malware",
            "threshold": result["bucket_window_detection"]["60"]["val_at_chosen"]["threshold"],
            "selected_on": "val",
            "val_metrics": result["bucket_window_detection"]["60"]["val_at_chosen"],
            "test_metrics": result["bucket_window_detection"]["60"]["test_at_chosen"],
            "val_passed_requirements": True,
            "test_passed_requirements": False,
            "enabled": False,
        }
    ]
