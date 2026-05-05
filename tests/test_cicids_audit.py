from pathlib import Path
import zipfile

import pandas as pd

from traffic_bert.data.cicids_audit import (
    CicidsCsvSpec,
    build_cicids_coverage_audit,
    label_distribution,
)
from traffic_bert.labels import LabelMap


def test_label_distribution_maps_source_major_and_minor_with_empty_labels() -> None:
    label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))
    labels = pd.Series(
        [
            " BENIGN ",
            "Web Attack \x96 Brute Force",
            "Web Attack \x96 XSS",
            "Web Attack \x96 Sql Injection",
            None,
            "",
        ]
    )

    result = label_distribution(labels, label_map)

    assert result["source_labels"]["BENIGN"] == 1
    assert result["major_labels"] == {"benign": 1, "web_attack": 3}
    assert result["minor_labels"] == {
        "web_bruteforce": 1,
        "xss": 1,
        "sql_injection": 1,
    }


def test_cicids_coverage_audit_reports_csv_and_pcap_presence(tmp_path: Path) -> None:
    raw_dir = tmp_path / "CICIDS2017"
    (raw_dir / "pcaps").mkdir(parents=True)
    (raw_dir / "labels").mkdir()
    (raw_dir / "pcaps" / "Friday-WorkingHours.pcap").write_bytes(b"pcap")
    (raw_dir / "labels" / "cicids2017_friday_ddos.csv").write_text("placeholder")

    csv_path = tmp_path / "ddos.csv"
    pd.DataFrame(
        [
            {" Source IP": "10.0.0.1", " Label": "BENIGN"},
            {" Source IP": "10.0.0.2", " Label": "DDoS"},
            {" Source IP": None, " Label": None},
        ]
    ).to_csv(csv_path, index=False)
    label_zip = raw_dir / "csvs" / "GeneratedLabelledFlows.zip"
    label_zip.parent.mkdir()
    with zipfile.ZipFile(label_zip, "w") as archive:
        archive.write(csv_path, "TrafficLabelling /Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

    audit = build_cicids_coverage_audit(
        raw_dir=raw_dir,
        label_zip=label_zip,
        label_map_path=Path("configs/label_map.yaml"),
        jobs=(
            CicidsCsvSpec(
                "friday_ddos",
                "Friday-WorkingHours.pcap",
                ("Friday", "DDos"),
                ("DDoS",),
            ),
        ),
    )

    item = audit["items"][0]
    assert item["csv_processed"] is True
    assert item["pcap_exists"] is True
    assert item["rows"] == 3
    assert item["valid_label_rows"] == 2
    assert item["empty_label_rows"] == 1
    assert item["major_labels"] == {"benign": 1, "dos_ddos": 1}
