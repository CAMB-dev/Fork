import pandas as pd
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "add_flow_context.py"
SPEC = importlib.util.spec_from_file_location("add_flow_context", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
add_context = MODULE.add_context


def test_add_context_counts_prior_host_window_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "flow_id": "a",
                "source_file": "one.pcap",
                "endpoint_a": "10.0.0.1:1111",
                "endpoint_b": "10.0.0.2:80",
                "start_time": 100.0,
                "connection_type": "tcp_control_only",
                "payload_byte_length": 0,
            },
            {
                "flow_id": "b",
                "source_file": "one.pcap",
                "endpoint_a": "10.0.0.1:2222",
                "endpoint_b": "10.0.0.3:443",
                "start_time": 130.0,
                "connection_type": "tcp_reset_or_refused",
                "payload_byte_length": 0,
            },
            {
                "flow_id": "c",
                "source_file": "one.pcap",
                "endpoint_a": "10.0.0.1:3333",
                "endpoint_b": "10.0.0.4:443",
                "start_time": 170.0,
                "connection_type": "tcp_payload",
                "payload_byte_length": 10,
            },
        ]
    )

    output = add_context(frame)

    assert output.loc[0, "context_host_prev_60s_count"] == 0
    assert output.loc[1, "context_host_prev_60s_count"] == 1
    assert output.loc[1, "context_host_prev_60s_control_count"] == 1
    assert output.loc[2, "context_host_prev_60s_count"] == 1
    assert output.loc[2, "context_host_prev_300s_count"] == 2
    assert output.loc[2, "context_host_prev_60s_reset_count"] == 1
    assert output.loc[2, "context_host_prev_60s_unique_dst_ports"] == 1
