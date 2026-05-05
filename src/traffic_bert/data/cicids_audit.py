"""CICIDS2017 labelled-flow coverage audit helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from traffic_bert.config import write_json
from traffic_bert.labels import LabelMap


@dataclass(frozen=True)
class CicidsCsvSpec:
    slug: str
    pcap_name: str
    csv_contains: tuple[str, ...]
    attack_labels: tuple[str, ...] = ()


CICIDS2017_JOBS: tuple[CicidsCsvSpec, ...] = (
    CicidsCsvSpec("monday_benign", "Monday-WorkingHours.pcap", ("Monday",)),
    CicidsCsvSpec(
        "tuesday_patator",
        "Tuesday-WorkingHours.pcap",
        ("Tuesday",),
        ("FTP-Patator", "SSH-Patator"),
    ),
    CicidsCsvSpec(
        "wednesday_dos",
        "Wednesday-workingHours.pcap",
        ("Wednesday",),
        ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "Heartbleed"),
    ),
    CicidsCsvSpec(
        "thursday_web",
        "Thursday-WorkingHours.pcap",
        ("Thursday", "WebAttacks"),
        (
            "Web Attack \x96 Brute Force",
            "Web Attack \x96 XSS",
            "Web Attack \x96 Sql Injection",
        ),
    ),
    CicidsCsvSpec(
        "thursday_infiltration",
        "Thursday-WorkingHours.pcap",
        ("Thursday", "Infilteration"),
        ("Infiltration",),
    ),
    CicidsCsvSpec("friday_bot", "Friday-WorkingHours.pcap", ("Friday", "Morning"), ("Bot",)),
    CicidsCsvSpec(
        "friday_portscan",
        "Friday-WorkingHours.pcap",
        ("Friday", "PortScan"),
        ("PortScan",),
    ),
    CicidsCsvSpec("friday_ddos", "Friday-WorkingHours.pcap", ("Friday", "DDos"), ("DDoS",)),
)


def find_csv_name(label_zip: Path, filename_contains: tuple[str, ...]) -> str:
    needles = tuple(item.lower() for item in filename_contains)
    with zipfile.ZipFile(label_zip) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and all(needle in Path(name).name.lower() for needle in needles)
        ]
    if not matches:
        raise FileNotFoundError(f"no CSV containing {filename_contains!r} in {label_zip}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous CSVs for {filename_contains!r}: {matches}")
    return matches[0]


def read_label_csv(label_zip: Path, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(label_zip) as archive:
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, encoding="latin1", low_memory=False)


def cicids_column(frame: pd.DataFrame, name: str) -> str:
    for column in frame.columns:
        if column.strip().lower() == name:
            return column
    raise ValueError(f"CSV has no {name!r} column")


def clean_label_series(values: pd.Series) -> pd.Series:
    labels = values.dropna().astype(str).str.strip()
    return labels[labels != ""]


def label_distribution(labels: pd.Series, label_map: LabelMap) -> dict:
    source_counts = Counter(clean_label_series(labels))
    major_counts: Counter[str] = Counter()
    minor_counts: Counter[str] = Counter()
    coarse_fine: Counter[tuple[str, str]] = Counter()
    for source_label, count in source_counts.items():
        target = label_map.map_source_label(str(source_label))
        major_counts[target.major_label] += int(count)
        if target.minor_labels:
            for minor_label in target.minor_labels:
                minor_counts[minor_label] += int(count)
                coarse_fine[(target.major_label, minor_label)] += int(count)
        else:
            coarse_fine[(target.major_label, "")] += int(count)
    return {
        "source_labels": {str(key): int(value) for key, value in source_counts.items()},
        "major_labels": {str(key): int(value) for key, value in major_counts.items()},
        "minor_labels": {str(key): int(value) for key, value in minor_counts.items()},
        "coarse_fine": [
            {"major_label": major, "minor_label": minor, "rows": int(count)}
            for (major, minor), count in sorted(coarse_fine.items())
        ],
    }


def _processed_distribution(processed_path: Path) -> dict | None:
    if not processed_path.exists():
        return None
    available_columns = set(pq.read_schema(processed_path).names)
    columns = [
        "flow_id",
        "source_label",
        "major_label",
        "minor_labels",
        "has_payload",
    ]
    frame = pd.read_parquet(
        processed_path,
        columns=[column for column in columns if column in available_columns],
    )
    minor_counts: Counter[str] = Counter()
    if "minor_labels" in frame:
        for value in frame["minor_labels"]:
            if value is not None:
                minor_counts.update(str(item) for item in value)
    return {
        "path": str(processed_path),
        "rows": int(len(frame)),
        "flows": int(frame["flow_id"].nunique()) if "flow_id" in frame else int(len(frame)),
        "source_labels": dict(Counter(frame.get("source_label", []))),
        "major_labels": dict(Counter(frame.get("major_label", []))),
        "minor_labels": dict(minor_counts),
        "empty_payload_rows": int((~frame["has_payload"]).sum()) if "has_payload" in frame else None,
    }


def build_cicids_coverage_audit(
    *,
    raw_dir: Path,
    label_zip: Path,
    label_map_path: Path,
    processed_dir: Path | None = None,
    jobs: tuple[CicidsCsvSpec, ...] = CICIDS2017_JOBS,
) -> dict:
    label_map = LabelMap.from_yaml(label_map_path)
    items = []
    for job in jobs:
        csv_name = find_csv_name(label_zip, job.csv_contains)
        frame = read_label_csv(label_zip, csv_name)
        label_col = cicids_column(frame, "label")
        labels = clean_label_series(frame[label_col])
        distribution = label_distribution(frame[label_col], label_map)
        pcap_path = raw_dir / "pcaps" / job.pcap_name
        extracted_label_path = raw_dir / "labels" / f"cicids2017_{job.slug}.csv"
        processed_path = processed_dir / f"flows_{job.slug}.parquet" if processed_dir else None
        item = {
            "slug": job.slug,
            "csv_name": csv_name,
            "csv_processed": extracted_label_path.exists(),
            "extracted_label_path": str(extracted_label_path),
            "pcap_name": job.pcap_name,
            "pcap_path": str(pcap_path),
            "pcap_exists": pcap_path.exists(),
            "rows": int(len(frame)),
            "valid_label_rows": int(len(labels)),
            "empty_label_rows": int(len(frame) - len(labels)),
            "attack_labels": list(job.attack_labels),
            **distribution,
        }
        if processed_path is not None:
            processed_distribution = _processed_distribution(processed_path)
            item["processed_shard"] = processed_distribution or {
                "path": str(processed_path),
                "exists": False,
            }
        items.append(item)

    missing_pcaps = sorted({item["pcap_name"] for item in items if not item["pcap_exists"]})
    missing_csv_extracts = [
        item["extracted_label_path"] for item in items if not item["csv_processed"]
    ]
    return {
        "raw_dir": str(raw_dir),
        "label_zip": str(label_zip),
        "processed_dir": str(processed_dir) if processed_dir is not None else None,
        "csv_count": len(items),
        "pcap_count": len({item["pcap_name"] for item in items if item["pcap_exists"]}),
        "missing_pcaps": missing_pcaps,
        "missing_csv_extracts": missing_csv_extracts,
        "items": items,
    }


def write_cicids_coverage_artifacts(audit: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "label_audit.json", audit)

    source_rows = []
    coarse_fine_rows = []
    for item in audit["items"]:
        base = {
            "slug": item["slug"],
            "csv_name": item["csv_name"],
            "pcap_name": item["pcap_name"],
            "pcap_exists": item["pcap_exists"],
        }
        for source_label, rows in item["source_labels"].items():
            source_rows.append({**base, "source_label": source_label, "rows": rows})
        for row in item["coarse_fine"]:
            coarse_fine_rows.append({**base, **row})

    pd.DataFrame(source_rows).to_csv(output_dir / "source_label_distribution.csv", index=False)
    pd.DataFrame(coarse_fine_rows).to_csv(
        output_dir / "coarse_fine_distribution.csv",
        index=False,
    )
