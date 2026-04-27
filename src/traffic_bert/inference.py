"""Hierarchical inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from traffic_bert.labels import LabelMap


@dataclass(frozen=True)
class ActivatedMinorLabel:
    label: str
    prob: float


@dataclass(frozen=True)
class HierarchicalPrediction:
    major_label: str
    major_prob: float
    primary_minor_label: str | None
    primary_minor_prob: float | None
    activated_minor_labels: list[ActivatedMinorLabel]

    def as_dict(self) -> dict:
        return {
            "major_label": self.major_label,
            "major_prob": self.major_prob,
            "primary_minor_label": self.primary_minor_label,
            "primary_minor_prob": self.primary_minor_prob,
            "activated_minor_labels": [
                {"label": item.label, "prob": item.prob}
                for item in self.activated_minor_labels
            ],
        }


def decode_hierarchical_prediction(
    major_logits: torch.Tensor,
    minor_logits: torch.Tensor,
    label_map: LabelMap,
    thresholds: dict[str, float] | None = None,
) -> HierarchicalPrediction:
    thresholds = thresholds or label_map.thresholds(default=0.5)
    major_probs = torch.softmax(major_logits.detach().float(), dim=-1)
    minor_probs = torch.sigmoid(minor_logits.detach().float())

    major_idx = int(torch.argmax(major_probs).item())
    major_label = label_map.major_labels[major_idx]
    major_prob = float(major_probs[major_idx].item())

    active: list[ActivatedMinorLabel] = []
    for minor_label in label_map.minor_labels_by_major.get(major_label, []):
        minor_idx = label_map.minor_to_id.get(minor_label)
        if minor_idx is None:
            continue
        prob = float(minor_probs[minor_idx].item())
        if prob >= thresholds.get(minor_label, 0.5):
            active.append(ActivatedMinorLabel(label=minor_label, prob=prob))

    active.sort(key=lambda item: item.prob, reverse=True)
    primary = active[0] if active else None
    return HierarchicalPrediction(
        major_label=major_label,
        major_prob=major_prob,
        primary_minor_label=primary.label if primary else None,
        primary_minor_prob=primary.prob if primary else None,
        activated_minor_labels=active,
    )

