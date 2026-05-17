from traffic_bert.tokenizer import ByteTokenizer, PacketChunk


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    data = bytes([0x00, 0x48, 0xFF])
    tokens = tokenizer.bytes_to_tokens(data)

    assert tokens == ["b_00", "b_48", "b_ff"]
    assert tokenizer.tokens_to_bytes(tokens) == data


def test_parse_hex_and_window_flow() -> None:
    tokenizer = ByteTokenizer()
    data = tokenizer.parse_hex("0x474554")
    windows = tokenizer.encode_flow(
        [PacketChunk(direction="fwd", data=data)],
        max_length=8,
        stride=4,
        padding=True,
    )

    assert data == b"GET"
    assert len(windows) == 1
    assert len(windows[0].input_ids) == 8
    assert "[PKT_FWD]" in windows[0].tokens
    assert "[PKT_END]" in windows[0].tokens


def test_connection_type_prefix_token_keeps_byte_ids_stable() -> None:
    tokenizer = ByteTokenizer()
    byte_id = tokenizer.token_to_id["b_00"]
    windows = tokenizer.encode_flow(
        [PacketChunk(direction="fwd", data=b"A")],
        max_length=8,
        stride=4,
        padding=True,
        prefix_tokens=[tokenizer.connection_type_token("tcp_reset_or_refused")],
    )

    assert byte_id == 8
    assert windows[0].tokens[1] == "[CONN_TCP_RESET_OR_REFUSED]"
    assert "[PKT_FWD]" in windows[0].tokens


def test_long_flow_creates_multiple_windows() -> None:
    tokenizer = ByteTokenizer()
    tokens = tokenizer.bytes_to_tokens(bytes(range(20)))
    windows = tokenizer.window_tokens(tokens, max_length=10, stride=6)

    assert len(windows) == 3
    assert all(len(item.input_ids) == 10 for item in windows)
