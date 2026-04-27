"""Label taxonomy and source-label mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LabelTarget:
    major_label: str
    minor_labels: tuple[str, ...]


@dataclass(frozen=True)
class LabelMappingRule:
    pattern: str
    major_label: str
    minor_labels: tuple[str, ...]

    def matches(self, source_label: str) -> bool:
        return re.search(self.pattern, source_label, flags=re.IGNORECASE) is not None


class LabelMap:
    """Load and apply the unified attack taxonomy."""

    def __init__(
        self,
        major_labels: list[str],
        minor_labels_by_major: dict[str, list[str]],
        mappings: list[LabelMappingRule],
        fallback: LabelTarget,
    ) -> None:
        self.major_labels = major_labels
        self.minor_labels_by_major = minor_labels_by_major
        self.mappings = mappings
        self.fallback = fallback
        self.major_to_id = {label: idx for idx, label in enumerate(major_labels)}
        self.minor_labels = [
            minor
            for major in major_labels
            for minor in minor_labels_by_major.get(major, [])
        ]
        self.minor_to_id = {label: idx for idx, label in enumerate(self.minor_labels)}
        self.minor_to_major = {
            minor: major
            for major, minors in minor_labels_by_major.items()
            for minor in minors
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelMap":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        mappings = [
            LabelMappingRule(
                pattern=item["pattern"],
                major_label=item["major_label"],
                minor_labels=tuple(item.get("minor_labels", [])),
            )
            for item in raw.get("mappings", [])
        ]
        fallback_raw = raw.get("fallback", {"major_label": "other_attack", "minor_labels": []})
        return cls(
            major_labels=list(raw["major_labels"]),
            minor_labels_by_major={
                str(major): list(minors or [])
                for major, minors in raw.get("minor_labels", {}).items()
            },
            mappings=mappings,
            fallback=LabelTarget(
                major_label=fallback_raw["major_label"],
                minor_labels=tuple(fallback_raw.get("minor_labels", [])),
            ),
        )

    def map_source_label(self, source_label: str | None) -> LabelTarget:
        if not source_label:
            return self.fallback
        for rule in self.mappings:
            if rule.matches(source_label):
                return LabelTarget(rule.major_label, rule.minor_labels)
        return self.fallback

    def major_id(self, major_label: str) -> int:
        return self.major_to_id[major_label]

    def minor_multi_hot(self, minor_labels: list[str] | tuple[str, ...]) -> list[float]:
        values = [0.0] * len(self.minor_labels)
        for label in minor_labels:
            idx = self.minor_to_id.get(label)
            if idx is not None:
                values[idx] = 1.0
        return values

    def minor_major_ids(self) -> list[int]:
        return [
            self.major_id(self.minor_to_major[label])
            for label in self.minor_labels
        ]

    def thresholds(self, default: float = 0.5) -> dict[str, float]:
        return {label: default for label in self.minor_labels}

    def as_dict(self) -> dict[str, Any]:
        return {
            "major_labels": self.major_labels,
            "minor_labels": self.minor_labels,
            "minor_labels_by_major": self.minor_labels_by_major,
        }

