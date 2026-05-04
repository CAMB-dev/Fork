import pandas as pd

from traffic_bert.data.cic import (
    CicFlowLabelIndex,
    canonical_flow_key,
    normalize_lookup_timestamp,
    parse_cic_timestamp,
)


def test_canonical_flow_key_is_bidirectional() -> None:
    left = canonical_flow_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
    right = canonical_flow_key("10.0.0.2", "10.0.0.1", 80, 1234, "tcp")

    assert left == right


def test_canonical_flow_key_normalizes_float_protocol_values() -> None:
    tcp_int = canonical_flow_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
    tcp_float = canonical_flow_key("10.0.0.1", "10.0.0.2", 1234, 80, 6.0)
    udp_text_float = canonical_flow_key("10.0.0.1", "10.0.0.2", 1234, 53, "17.0")

    assert tcp_float == tcp_int
    assert udp_text_float[0] == "udp"


def test_cic_timestamp_parsing_prefers_day_first() -> None:
    assert str(parse_cic_timestamp("7/7/2017 15:30")) == "2017-07-07 15:30:00"
    assert str(parse_cic_timestamp("3/7/2017 8:59")) == "2017-07-03 08:59:00"
    assert str(normalize_lookup_timestamp(1499433540.0)) == "2017-07-07 13:19:00"


def test_cic_flow_label_index_lookup() -> None:
    frame = pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": 6,
                "Label": "DoS Hulk",
                "Timestamp": "2017-07-07 10:00:00",
            }
        ]
    )
    index = CicFlowLabelIndex.from_frame(frame)

    assert index.lookup("10.0.0.2", "10.0.0.1", 80, 1234, "tcp") == "DoS Hulk"


def test_cic_flow_label_index_uses_nearest_timestamp_for_duplicate_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": "6.0",
                "Label": "BENIGN",
                "Timestamp": "2017-07-07 09:00:00",
            },
            {
                "Source IP": "10.0.0.1",
                "Destination IP": "10.0.0.2",
                "Source Port": 1234,
                "Destination Port": 80,
                "Protocol": "6.0",
                "Label": "DDoS",
                "Timestamp": "2017-07-07 15:30:00",
            },
        ]
    )
    index = CicFlowLabelIndex.from_frame(frame)

    assert index.lookup("10.0.0.2", "10.0.0.1", 80, 1234, "tcp", "2017-07-07 15:31:00") == "DDoS"
