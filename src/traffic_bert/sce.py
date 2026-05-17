"""Minimal Semantic Conversion Encoder codebook utilities.

This module intentionally starts with a conservative, deterministic codebook:
frequent byte chunks become discrete SCE tokens and all unmatched bytes fall
back to the existing byte tokenizer. It is a bridge toward the SCE route, not a
replacement for the current Byte-BERT baseline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from traffic_bert.tokenizer import ByteTokenizer


SCE_TOKEN_PREFIX = "[SCE_"


@dataclass(frozen=True)
class SemanticCodebookEntry:
    token: str
    pattern_hex: str
    count: int
    score: float | None = None
    label: str | None = None

    @property
    def pattern(self) -> bytes:
        return bytes.fromhex(self.pattern_hex)


@dataclass(frozen=True)
class SemanticCodebook:
    chunk_size: int
    entries: tuple[SemanticCodebookEntry, ...]
    kind: str = "frequency_byte_chunk"
    version: int = 1
    drop_zero_chunks: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        tokens = [entry.token for entry in self.entries]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate SCE tokens in codebook")
        patterns = [entry.pattern_hex for entry in self.entries]
        if len(patterns) != len(set(patterns)):
            raise ValueError("duplicate byte patterns in codebook")

    @property
    def tokens(self) -> list[str]:
        return [entry.token for entry in self.entries]

    @property
    def pattern_to_token(self) -> dict[bytes, str]:
        return {entry.pattern: entry.token for entry in self.entries}

    def encode_bytes(
        self,
        data: bytes,
        *,
        tokenizer: ByteTokenizer | None = None,
        fallback_to_bytes: bool = True,
    ) -> list[str]:
        tokenizer = tokenizer or ByteTokenizer(extra_tokens=self.tokens)
        mapping = self.pattern_to_token
        output: list[str] = []
        for chunk in iter_byte_chunks(data, self.chunk_size):
            token = mapping.get(chunk)
            if token is not None:
                output.append(token)
            elif fallback_to_bytes:
                output.extend(tokenizer.bytes_to_tokens(chunk))
        return output

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "kind": self.kind,
            "chunk_size": self.chunk_size,
            "drop_zero_chunks": self.drop_zero_chunks,
            "entries": [
                {
                    "token": entry.token,
                    "pattern_hex": entry.pattern_hex,
                    "count": int(entry.count),
                    **({} if entry.score is None else {"score": float(entry.score)}),
                    **({} if entry.label is None else {"label": entry.label}),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SemanticCodebook":
        return cls(
            version=int(payload.get("version", 1)),
            kind=str(payload.get("kind", "frequency_byte_chunk")),
            chunk_size=int(payload["chunk_size"]),
            drop_zero_chunks=bool(payload.get("drop_zero_chunks", False)),
            entries=tuple(
                SemanticCodebookEntry(
                    token=str(entry["token"]),
                    pattern_hex=str(entry["pattern_hex"]),
                    count=int(entry.get("count", 0)),
                    score=float(entry["score"]) if entry.get("score") is not None else None,
                    label=str(entry["label"]) if entry.get("label") is not None else None,
                )
                for entry in payload.get("entries", [])
            ),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticCodebook":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def iter_byte_chunks(data: bytes, chunk_size: int) -> Iterable[bytes]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        if chunk:
            yield chunk


def _payload_to_bytes(payload: object) -> bytes:
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, bytes):
        return payload
    return bytes(payload)


def _valid_chunk(chunk: bytes, *, drop_zero_chunks: bool) -> bool:
    return not (drop_zero_chunks and all(value == 0 for value in chunk))


def learn_frequency_codebook(
    payloads: Iterable[bytes],
    *,
    chunk_size: int = 4,
    max_entries: int = 256,
    min_count: int = 2,
    drop_zero_chunks: bool = True,
) -> SemanticCodebook:
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    counter: Counter[bytes] = Counter()
    for payload in payloads:
        payload = _payload_to_bytes(payload)
        for chunk in iter_byte_chunks(payload, chunk_size):
            if not _valid_chunk(chunk, drop_zero_chunks=drop_zero_chunks):
                continue
            counter[chunk] += 1
    ranked = sorted(
        ((chunk, count) for chunk, count in counter.items() if count >= min_count),
        key=lambda item: (-item[1], item[0]),
    )[:max_entries]
    entries = tuple(
        SemanticCodebookEntry(
            token=f"{SCE_TOKEN_PREFIX}{index:04d}]",
            pattern_hex=chunk.hex(),
            count=int(count),
        )
        for index, (chunk, count) in enumerate(ranked)
    )
    return SemanticCodebook(
        chunk_size=chunk_size,
        entries=entries,
        drop_zero_chunks=drop_zero_chunks,
    )


def learn_label_lift_codebook(
    payloads: Iterable[bytes],
    labels: Iterable[str],
    *,
    chunk_size: int = 4,
    max_entries: int = 256,
    min_count: int = 2,
    drop_zero_chunks: bool = True,
    target_label: str | None = None,
) -> SemanticCodebook:
    """Learn chunks by label association instead of raw frequency.

    The score is the largest absolute difference between P(label | chunk) and
    the global P(label). With ``target_label`` set, only that label is scored.
    Rows count each chunk once, which prevents long flows from dominating.
    """

    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    chunk_rows: Counter[bytes] = Counter()
    label_rows: Counter[str] = Counter()
    chunk_label_rows: dict[bytes, Counter[str]] = defaultdict(Counter)
    total_rows = 0
    for payload, raw_label in zip(payloads, labels, strict=True):
        label = str(raw_label)
        label_rows[label] += 1
        total_rows += 1
        payload = _payload_to_bytes(payload)
        chunks = {
            chunk
            for chunk in iter_byte_chunks(payload, chunk_size)
            if _valid_chunk(chunk, drop_zero_chunks=drop_zero_chunks)
        }
        for chunk in chunks:
            chunk_rows[chunk] += 1
            chunk_label_rows[chunk][label] += 1

    if total_rows == 0:
        return SemanticCodebook(
            kind="label_lift_byte_chunk",
            chunk_size=chunk_size,
            entries=(),
            drop_zero_chunks=drop_zero_chunks,
        )

    labels_to_score = [target_label] if target_label is not None else list(label_rows)
    scored: list[tuple[bytes, int, float, str]] = []
    for chunk, count in chunk_rows.items():
        if count < min_count:
            continue
        best_score = -1.0
        best_label = ""
        for label in labels_to_score:
            global_rate = label_rows[label] / total_rows
            chunk_rate = chunk_label_rows[chunk][label] / count
            score = (
                chunk_rate - global_rate
                if target_label is not None
                else abs(chunk_rate - global_rate)
            )
            if score > best_score:
                best_score = score
                best_label = str(label)
        if target_label is not None and best_score <= 0:
            continue
        scored.append((chunk, count, best_score, best_label))

    ranked = sorted(scored, key=lambda item: (-item[2], -item[1], item[0]))[:max_entries]
    entries = tuple(
        SemanticCodebookEntry(
            token=f"{SCE_TOKEN_PREFIX}{index:04d}]",
            pattern_hex=chunk.hex(),
            count=int(count),
            score=float(score),
            label=label,
        )
        for index, (chunk, count, score, label) in enumerate(ranked)
    )
    return SemanticCodebook(
        kind="label_lift_byte_chunk",
        chunk_size=chunk_size,
        entries=entries,
        drop_zero_chunks=drop_zero_chunks,
    )
