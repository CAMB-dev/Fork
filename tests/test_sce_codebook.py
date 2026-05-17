from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from traffic_bert.cli import app
from traffic_bert.sce import SemanticCodebook, learn_frequency_codebook, learn_label_lift_codebook
from traffic_bert.tokenizer import ByteTokenizer


def test_frequency_codebook_encodes_known_chunks_with_byte_fallback(tmp_path: Path) -> None:
    codebook = learn_frequency_codebook(
        [b"GET /index", b"GET /login"],
        chunk_size=2,
        max_entries=2,
        min_count=2,
    )
    assert [entry.pattern for entry in codebook.entries] == [b"GE", b"T "]

    tokenizer = ByteTokenizer(extra_tokens=codebook.tokens)
    tokens = codebook.encode_bytes(b"GET /x", tokenizer=tokenizer)

    assert codebook.entries[0].token in tokens
    assert codebook.entries[1].token in tokens
    assert "b_78" in tokens
    assert tokenizer.tokens_to_ids(tokens)[0] != tokenizer.unk_token_id

    path = tmp_path / "sce_codebook.json"
    codebook.save(path)
    loaded = SemanticCodebook.load(path)
    assert loaded.to_dict() == codebook.to_dict()


def test_frequency_codebook_drops_zero_chunks_by_default() -> None:
    codebook = learn_frequency_codebook(
        [b"\x00\x00AB", b"\x00\x00AB"],
        chunk_size=2,
        max_entries=4,
        min_count=1,
    )

    assert [entry.pattern for entry in codebook.entries] == [b"AB"]
    assert codebook.drop_zero_chunks is True


def test_label_lift_codebook_prefers_label_specific_chunks() -> None:
    codebook = learn_label_lift_codebook(
        [b"AAAA", b"AAAA", b"BBBB", b"BBBB"],
        ["benign", "benign", "botnet_malware", "botnet_malware"],
        chunk_size=2,
        max_entries=2,
        min_count=1,
        target_label="botnet_malware",
    )

    assert codebook.kind == "label_lift_byte_chunk"
    assert codebook.entries[0].pattern == b"BB"
    assert codebook.entries[0].label == "botnet_malware"
    assert codebook.entries[0].score == 0.5


def test_sce_build_codebook_cli_writes_json(tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "codebook.json"
    pd.DataFrame(
        [
            {"view": "masked_header_packet", "bytes": b"GET /index"},
            {"view": "masked_header_packet", "bytes": b"GET /login"},
            {"view": "payload_only", "bytes": b"POST /"},
        ]
    ).to_parquet(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "sce",
            "build-codebook",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--view",
            "masked_header_packet",
            "--chunk-size",
            "2",
            "--max-entries",
            "4",
            "--min-count",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "frequency_byte_chunk"
    assert payload["chunk_size"] == 2
    assert payload["drop_zero_chunks"] is True
    assert len(payload["entries"]) >= 1


def test_sce_build_label_lift_codebook_cli_writes_scores(tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "codebook.json"
    pd.DataFrame(
        [
            {"view": "masked_header_packet", "major_label": "benign", "bytes": b"AAAA"},
            {"view": "masked_header_packet", "major_label": "benign", "bytes": b"AAAA"},
            {
                "view": "masked_header_packet",
                "major_label": "botnet_malware",
                "bytes": b"BBBB",
            },
            {
                "view": "masked_header_packet",
                "major_label": "botnet_malware",
                "bytes": b"BBBB",
            },
        ]
    ).to_parquet(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "sce",
            "build-codebook",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--view",
            "masked_header_packet",
            "--ranking",
            "label_lift",
            "--target-label",
            "botnet_malware",
            "--chunk-size",
            "2",
            "--max-entries",
            "2",
            "--min-count",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "label_lift_byte_chunk"
    assert payload["entries"][0]["pattern_hex"] == "4242"
    assert payload["entries"][0]["label"] == "botnet_malware"
    assert payload["entries"][0]["score"] == 0.5


def test_sce_build_label_lift_codebook_cli_can_sample_per_label(tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "codebook.json"
    pd.DataFrame(
        [
            *(
                {"view": "masked_header_packet", "major_label": "benign", "bytes": b"AAAA"}
                for _ in range(10)
            ),
            *(
                {
                    "view": "masked_header_packet",
                    "major_label": "botnet_malware",
                    "bytes": b"BBBB",
                }
                for _ in range(3)
            ),
        ]
    ).to_parquet(input_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "sce",
            "build-codebook",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--view",
            "masked_header_packet",
            "--ranking",
            "label_lift",
            "--target-label",
            "botnet_malware",
            "--chunk-size",
            "2",
            "--max-entries",
            "2",
            "--min-count",
            "1",
            "--max-rows-per-label",
            "2",
            "--sample-seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    stdout = json.loads(result.stdout)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout["rows_used"] == 4
    assert stdout["max_rows_per_label"] == 2
    assert payload["entries"][0]["pattern_hex"] == "4242"
