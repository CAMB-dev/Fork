"""Byte-level tokenizer used to turn traffic bytes into BERT token ids."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
PKT_FWD_TOKEN = "[PKT_FWD]"
PKT_BWD_TOKEN = "[PKT_BWD]"
PKT_END_TOKEN = "[PKT_END]"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
    PKT_FWD_TOKEN,
    PKT_BWD_TOKEN,
    PKT_END_TOKEN,
]

BYTE_TOKENS = [f"b_{value:02x}" for value in range(256)]
HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]+$")


@dataclass(frozen=True)
class PacketChunk:
    """One packet's selected bytes plus direction within a flow."""

    direction: str
    data: bytes


@dataclass(frozen=True)
class WindowEncoding:
    """A BERT-ready window cut from a flow token sequence."""

    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[str]


class ByteTokenizer:
    """Fixed-vocabulary byte tokenizer.

    The tokenizer maps each raw byte to a stable textual token, for example
    ``0x4f -> b_4f``. This preserves payload information while making the input
    compatible with BERT-style embedding tables and MLM pretraining.
    """

    def __init__(self) -> None:
        self.tokens = SPECIAL_TOKENS + BYTE_TOKENS
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id[SEP_TOKEN]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[MASK_TOKEN]

    def byte_to_token(self, value: int) -> str:
        if not 0 <= value <= 255:
            raise ValueError(f"byte value out of range: {value}")
        return f"b_{value:02x}"

    def token_to_byte(self, token: str) -> int | None:
        if len(token) == 4 and token.startswith("b_"):
            try:
                return int(token[2:], 16)
            except ValueError:
                return None
        return None

    def bytes_to_tokens(self, data: bytes) -> list[str]:
        return [self.byte_to_token(value) for value in data]

    def tokens_to_bytes(self, tokens: Iterable[str]) -> bytes:
        values: list[int] = []
        for token in tokens:
            value = self.token_to_byte(token)
            if value is not None:
                values.append(value)
        return bytes(values)

    def tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token.get(int(idx), UNK_TOKEN) for idx in ids]

    def flow_tokens(self, chunks: Sequence[PacketChunk]) -> list[str]:
        tokens: list[str] = []
        for chunk in chunks:
            direction = chunk.direction.lower()
            if direction in {"fwd", "forward", "client"}:
                tokens.append(PKT_FWD_TOKEN)
            elif direction in {"bwd", "backward", "server"}:
                tokens.append(PKT_BWD_TOKEN)
            else:
                raise ValueError(f"unsupported packet direction: {chunk.direction}")
            tokens.extend(self.bytes_to_tokens(chunk.data))
            tokens.append(PKT_END_TOKEN)
        return tokens

    def encode_tokens(
        self,
        tokens: Sequence[str],
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> WindowEncoding:
        sequence = [CLS_TOKEN, *tokens, SEP_TOKEN]
        if max_length is not None and len(sequence) > max_length:
            if not truncation:
                raise ValueError(
                    f"sequence length {len(sequence)} exceeds max_length={max_length}"
                )
            sequence = sequence[: max_length - 1] + [SEP_TOKEN]

        attention_mask = [1] * len(sequence)
        if max_length is not None and padding and len(sequence) < max_length:
            pad_count = max_length - len(sequence)
            sequence = [*sequence, *([PAD_TOKEN] * pad_count)]
            attention_mask.extend([0] * pad_count)

        return WindowEncoding(
            input_ids=self.tokens_to_ids(sequence),
            attention_mask=attention_mask,
            tokens=sequence,
        )

    def encode_bytes(
        self,
        data: bytes,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> WindowEncoding:
        return self.encode_tokens(
            self.bytes_to_tokens(data),
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )

    def window_tokens(
        self,
        tokens: Sequence[str],
        max_length: int = 512,
        stride: int = 384,
        padding: bool = True,
    ) -> list[WindowEncoding]:
        if max_length < 3:
            raise ValueError("max_length must leave room for [CLS], content, [SEP]")
        if stride <= 0:
            raise ValueError("stride must be positive")

        content_length = max_length - 2
        if not tokens:
            return [self.encode_tokens([], max_length=max_length, padding=padding)]

        windows: list[WindowEncoding] = []
        start = 0
        while start < len(tokens):
            end = min(start + content_length, len(tokens))
            windows.append(
                self.encode_tokens(tokens[start:end], max_length=max_length, padding=padding)
            )
            if end == len(tokens):
                break
            start += stride
        return windows

    def encode_flow(
        self,
        chunks: Sequence[PacketChunk],
        max_length: int = 512,
        stride: int = 384,
        padding: bool = True,
    ) -> list[WindowEncoding]:
        return self.window_tokens(
            self.flow_tokens(chunks),
            max_length=max_length,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def parse_hex(payload_hex: str) -> bytes:
        text = payload_hex.strip().replace(" ", "").replace("\n", "").replace("\t", "")
        if text.startswith(("0x", "0X")):
            text = text[2:]
        if not text:
            return b""
        if len(text) % 2 != 0:
            raise ValueError("hex payload must contain an even number of hex digits")
        if not HEX_RE.match(text):
            raise ValueError("hex payload contains non-hex characters")
        return bytes.fromhex(text)

    def save_vocab(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for token in self.tokens:
                handle.write(f"{token}\n")

