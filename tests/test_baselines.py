import pandas as pd

from traffic_bert.baselines import (
    bytes_to_hex_text,
    evaluate_major_baseline,
    train_major_baseline,
)
from traffic_bert.labels import LabelMap


def test_bytes_to_hex_text() -> None:
    assert bytes_to_hex_text(b"\x00A\xff") == "00 41 ff"


def test_tfidf_major_baseline_train_eval() -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    frame = pd.DataFrame(
        [
            {"bytes": b"normal http", "major_label": "benign"},
            {"bytes": b"normal dns", "major_label": "benign"},
            {"bytes": b"ddos flood", "major_label": "dos_ddos"},
            {"bytes": b"ddos attack", "major_label": "dos_ddos"},
        ]
    )

    pipeline = train_major_baseline(
        frame,
        label_map=label_map,
        model="logreg",
        ngram_range=(1, 2),
        max_features=100,
    )
    result = evaluate_major_baseline(pipeline, frame, label_map)

    assert "macro avg" in result.metrics

