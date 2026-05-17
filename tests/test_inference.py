import torch

from traffic_bert.inference import decode_hierarchical_prediction
from traffic_bert.labels import LabelMap


def test_decode_hierarchical_prediction_primary_and_candidates() -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    major_logits = torch.full((len(label_map.major_labels),), -5.0)
    major_logits[label_map.major_id("web_attack")] = 5.0
    minor_logits = torch.full((len(label_map.minor_labels),), -5.0)
    minor_logits[label_map.minor_to_id["sql_injection"]] = 3.0
    minor_logits[label_map.minor_to_id["xss"]] = 2.0

    prediction = decode_hierarchical_prediction(
        major_logits,
        minor_logits,
        label_map,
        thresholds={"sql_injection": 0.5, "xss": 0.5, "web_bruteforce": 0.5},
    )

    assert prediction.major_label == "web_attack"
    assert prediction.major_probs["web_attack"] == prediction.major_prob
    assert prediction.primary_minor_label == "sql_injection"
    assert [item.label for item in prediction.activated_minor_labels] == [
        "sql_injection",
        "xss",
    ]


def test_decode_hierarchical_prediction_major_only() -> None:
    label_map = LabelMap.from_yaml("configs/label_map.yaml")
    major_logits = torch.full((len(label_map.major_labels),), -5.0)
    major_logits[label_map.major_id("web_attack")] = 5.0
    minor_logits = torch.full((len(label_map.minor_labels),), -5.0)

    prediction = decode_hierarchical_prediction(major_logits, minor_logits, label_map)

    assert prediction.major_label == "web_attack"
    assert set(prediction.major_probs) == set(label_map.major_labels)
    assert prediction.primary_minor_label is None
    assert prediction.activated_minor_labels == []
