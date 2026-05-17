"""Write CICIDS2017 labelled-flow coverage audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_bert.data.cicids_audit import (
    build_cicids_coverage_audit,
    write_cicids_coverage_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/CICIDS2017"))
    parser.add_argument(
        "--label-zip",
        type=Path,
        default=Path("data/raw/CICIDS2017/csvs/GeneratedLabelledFlows.zip"),
    )
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cicids2017_coverage_masked_header_cap32_notcpclose"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_cicids_coverage_audit(
        raw_dir=args.raw_dir,
        label_zip=args.label_zip,
        label_map_path=args.label_map,
        processed_dir=args.processed_dir,
    )
    write_cicids_coverage_artifacts(audit, args.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
