from pathlib import Path
import importlib.util

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_processed_split.py"
SPEC = importlib.util.spec_from_file_location("audit_processed_split", SCRIPT_PATH)
assert SPEC is not None
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_module)
audit = audit_module.audit


def _write_split(path: Path, rows: int) -> None:
    pd.DataFrame(
        {
            "flow_id": [f"{path.stem}-{idx}" for idx in range(rows)],
            "source_file": ["friday.pcap"] * rows,
            "source_label": ["BENIGN"] * rows,
            "major_label": ["benign"] * rows,
            "minor_labels": [[] for _ in range(rows)],
            "view": ["payload_only"] * rows,
            "packet_count": [2] * rows,
            "payload_byte_length": [128] * rows,
        }
    ).to_parquet(path, index=False)


def test_audit_warns_when_split_contains_only_one_major_label(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    _write_split(train_path, 3)
    _write_split(val_path, 2)
    _write_split(test_path, 2)

    result = audit(train_path, val_path, test_path)

    assert result["all_major_labels"] == ["benign"]
    assert "fewer than 2 major labels across all splits: benign" in result["warnings"]
