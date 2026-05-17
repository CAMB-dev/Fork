from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pcap_demo_server.py"
SPEC = importlib.util.spec_from_file_location("pcap_demo_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pcap_demo_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pcap_demo_server)

_build_predict_command = pcap_demo_server._build_predict_command
_is_allowed_upload = pcap_demo_server._is_allowed_upload
_safe_filename = pcap_demo_server._safe_filename


def test_demo_upload_filename_validation() -> None:
    assert _is_allowed_upload("sample.pcap")
    assert _is_allowed_upload("sample.PCAPNG")
    assert not _is_allowed_upload("sample.txt")
    assert _safe_filename("../bad name.pcap") == "bad_name.pcap"


def test_demo_build_predict_command() -> None:
    command = _build_predict_command(
        cli_command="traffic-bert",
        pcap_path=Path("upload.pcap"),
        checkpoint=Path("model.pt"),
        summary_output=Path("summary.json"),
        flow_output=Path("flows.jsonl"),
        device="auto",
        max_windows=2,
        risk_threshold=0.7,
        top_flows=5,
        host_window_policy_path=Path("policy.json"),
    )

    assert command[:3] == ["traffic-bert", "predict", "pcap"]
    assert "--close-on-tcp-flags" not in command
    assert command[command.index("--host-window-policy-path") + 1] == "policy.json"
    assert command[command.index("--checkpoint") + 1] == "model.pt"
    assert command[command.index("--summary-output") + 1] == "summary.json"
