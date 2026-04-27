"""Validation checks for processed Parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "flow_id",
    "source_dataset",
    "source_file",
    "source_label",
    "major_label",
    "minor_labels",
    "split",
    "view",
    "packet_count",
    "payload_byte_length",
    "packet_byte_length",
    "has_payload",
    "packet_directions",
    "packet_lengths",
    "bytes",
}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }


def validate_processed_frame(frame: pd.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_columns",
                message=f"missing required columns: {', '.join(missing)}",
            )
        )
        return issues

    if frame.empty:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="empty_dataset",
                message="processed dataset contains no rows",
            )
        )
        return issues

    null_labels = int(frame["major_label"].isna().sum())
    if null_labels:
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_major_label",
                message=f"{null_labels} rows have missing major_label",
            )
        )

    bad_packet_count = int((frame["packet_count"] <= 0).sum())
    if bad_packet_count:
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_positive_packet_count",
                message=f"{bad_packet_count} rows have packet_count <= 0",
            )
        )

    empty_payload = int((~frame["has_payload"]).sum())
    if empty_payload:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="empty_payload",
                message=f"{empty_payload} rows have no L4 payload",
            )
        )

    duplicate_rows = int(frame.duplicated(subset=["flow_id", "view"]).sum())
    if duplicate_rows:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="duplicate_flow_view",
                message=f"{duplicate_rows} duplicate flow_id/view rows found",
            )
        )

    direction_length_mismatch = 0
    for _, row in frame.iterrows():
        if len(row["packet_directions"]) != len(row["packet_lengths"]):
            direction_length_mismatch += 1
    if direction_length_mismatch:
        issues.append(
            ValidationIssue(
                severity="error",
                code="packet_metadata_mismatch",
                message=f"{direction_length_mismatch} rows have mismatched packet metadata",
            )
        )

    return issues


def validation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    issues = validate_processed_frame(frame)
    return {
        "ok": not any(issue.severity == "error" for issue in issues),
        "issues": [issue.as_dict() for issue in issues],
    }

