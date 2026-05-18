from __future__ import annotations

import argparse
import cgi
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch

from traffic_bert.cli import (
    _checkpoint_uses_connection_tokens,
    _load_classifier,
    _load_host_window_policies,
    _load_thresholds,
    _resolve_semantic_codebook,
    _summarize_pcap_predictions,
    resolve_device,
)
from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.schema import InputView
from traffic_bert.inference import decode_hierarchical_prediction
from traffic_bert.labels import LabelMap
from traffic_bert.tokenizer import ByteTokenizer, PacketChunk


ALLOWED_SUFFIXES = {".pcap", ".pcapng"}


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Traffic Byte-BERT PCAP Demo</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #f6f8fb; color: #18212f; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 22px; }
    h1 { font-size: 24px; margin: 0; letter-spacing: 0; }
    .status { font-size: 13px; color: #526070; }
    .panel { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; padding: 18px; }
    form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
    input[type=file] { padding: 10px; border: 1px solid #ccd6e3; border-radius: 6px; background: #fff; }
    button { border: 0; border-radius: 6px; padding: 11px 16px; background: #1f5eff; color: white; font-weight: 650; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
    .metric { background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; padding: 14px; }
    .metric span { display: block; font-size: 12px; color: #66758a; margin-bottom: 7px; }
    .metric strong { font-size: 24px; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d9e1ec; border-radius: 8px; overflow: hidden; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #edf1f6; font-size: 13px; vertical-align: top; }
    th { background: #eef3f9; color: #405066; font-size: 12px; text-transform: uppercase; }
    .label { font-weight: 700; }
    .benign { color: #247a3b; }
    .attack { color: #b52020; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0f1724; color: #dbe7ff; border-radius: 8px; padding: 14px; max-height: 340px; overflow: auto; }
    @media (max-width: 760px) { main { padding: 16px; } .grid { grid-template-columns: 1fr 1fr; } header { display: block; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Traffic Byte-BERT PCAP 推理</h1>
    <div class="status" id="status">等待上传</div>
  </header>
  <section class="panel">
    <form id="upload-form">
      <input id="pcap" name="pcap" type="file" accept=".pcap,.pcapng" required />
      <button id="submit" type="submit">上传并推理</button>
    </form>
  </section>
  <section class="grid" id="metrics" hidden></section>
  <section id="hosts" hidden>
    <table>
      <thead><tr><th>Host</th><th>Flows</th><th>Predicted attacks</th><th>High risk</th><th>Max confidence</th></tr></thead>
      <tbody id="host-rows"></tbody>
    </table>
  </section>
  <section id="windows" hidden>
    <table>
      <thead><tr><th>Window</th><th>Host</th><th>Flows</th><th>Predicted attacks</th><th>High risk</th><th>Max confidence</th></tr></thead>
      <tbody id="window-rows"></tbody>
    </table>
  </section>
  <section id="results" hidden>
    <table>
      <thead><tr><th>Flow</th><th>预测</th><th>置信度</th><th>连接</th><th>Endpoints</th></tr></thead>
      <tbody id="flow-rows"></tbody>
    </table>
  </section>
  <section id="raw-section" hidden>
    <h2>原始摘要</h2>
    <pre id="raw"></pre>
  </section>
</main>
<script>
const form = document.getElementById('upload-form');
const submit = document.getElementById('submit');
const statusNode = document.getElementById('status');
const metrics = document.getElementById('metrics');
const hosts = document.getElementById('hosts');
const hostRows = document.getElementById('host-rows');
const windows = document.getElementById('windows');
const windowRows = document.getElementById('window-rows');
const results = document.getElementById('results');
const rows = document.getElementById('flow-rows');
const rawSection = document.getElementById('raw-section');
const raw = document.getElementById('raw');

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = document.getElementById('pcap').files[0];
  if (!file) return;
  submit.disabled = true;
  statusNode.textContent = '推理中...';
  metrics.hidden = true;
  hosts.hidden = true;
  windows.hidden = true;
  results.hidden = true;
  rawSection.hidden = true;
  const body = new FormData();
  body.append('pcap', file);
  try {
    const response = await fetch('/api/predict', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'prediction failed');
    const s = payload.summary;
    metrics.innerHTML = [
      metric('Flows', s.flow_count),
      metric('Predicted attacks', s.predicted_attack_flow_count),
      metric('High risk', s.high_risk_flow_count),
      metric('Elapsed seconds', payload.elapsed_seconds.toFixed(2))
    ].join('');
    rows.innerHTML = (payload.flows || []).slice(0, 100).map((flow) => {
      const isAttack = flow.major_label !== 'benign';
      const cls = isAttack ? 'attack' : 'benign';
      const prob = Number(flow.major_prob || 0).toFixed(4);
      return `<tr>
        <td>${flow.flow_id || ''}</td>
        <td class="label ${cls}">${flow.major_label || 'unknown'}</td>
        <td>${prob}</td>
        <td>${flow.protocol || ''}<br>${flow.connection_type || ''}<br>${flow.packet_count || 0} packets</td>
        <td>${flow.endpoint_a || ''}<br>${flow.endpoint_b || ''}</td>
      </tr>`;
    }).join('');
    hostRows.innerHTML = (s.top_host_risks || []).slice(0, 20).map((host) => {
      return `<tr>
        <td>${host.host || 'unknown'}</td>
        <td>${host.flow_count || 0}</td>
        <td>${host.predicted_attack_flow_count || 0}</td>
        <td>${host.high_risk_flow_count || 0}</td>
        <td>${Number(host.max_major_prob || 0).toFixed(4)}</td>
      </tr>`;
    }).join('');
    const policyWindows = Object.entries(s.policy_host_windows || {}).flatMap(([policy, items]) => {
      return (items || []).slice(0, 10).map((item) => ({ policy, ...item }));
    });
    const hostWindows = Object.entries(s.top_host_windows || {}).flatMap(([seconds, items]) => {
      return (items || []).slice(0, 10).map((item) => ({ policy: `${seconds}s`, ...item }));
    });
    windowRows.innerHTML = [...policyWindows, ...hostWindows].slice(0, 20).map((item) => {
      const start = Number(item.window_start_time || 0).toFixed(0);
      const attackCount = item.predicted_attack_flow_count ?? item.target_argmax_flow_count ?? 0;
      const hitCount = item.high_risk_flow_count ?? item.policy_hit_flow_count ?? 0;
      const probability = item.max_major_prob ?? item.max_target_prob ?? 0;
      return `<tr>
        <td>${item.policy}<br>${start}</td>
        <td>${item.host || 'unknown'}</td>
        <td>${item.flow_count || 0}</td>
        <td>${attackCount}</td>
        <td>${hitCount}</td>
        <td>${Number(probability || 0).toFixed(4)}</td>
      </tr>`;
    }).join('');
    raw.textContent = JSON.stringify({ summary: s }, null, 2);
    metrics.hidden = false;
    hosts.hidden = false;
    windows.hidden = false;
    results.hidden = false;
    rawSection.hidden = false;
    statusNode.textContent = '完成';
  } catch (error) {
    statusNode.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});
</script>
</body>
</html>
"""


def _safe_filename(value: str) -> str:
    name = Path(value or "upload.pcap").name
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)


def _is_allowed_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_SUFFIXES


def _build_predict_command(
    *,
    cli_command: str,
    pcap_path: Path,
    checkpoint: Path,
    summary_output: Path,
    flow_output: Path,
    device: str,
    max_windows: int,
    risk_threshold: float,
    top_flows: int,
    host_window_policy_path: Path | None = None,
) -> list[str]:
    command = [
        cli_command,
        "predict",
        "pcap",
        str(pcap_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--max-windows",
        str(max_windows),
        "--summary-output",
        str(summary_output),
        "--output",
        str(flow_output),
        "--risk-threshold",
        str(risk_threshold),
        "--top-flows",
        str(top_flows),
    ]
    if host_window_policy_path is not None:
        command.extend(["--host-window-policy-path", str(host_window_policy_path)])
    return command


def _read_jsonl(path: Path, limit: int = 1000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if len(items) >= limit:
                break
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _predict_pcap(server: "DemoServer", pcap_path: Path) -> dict[str, Any]:
    flows = PcapFlowExtractor(
        max_packets_per_flow=server.max_packets_per_flow,
        flow_timeout_seconds=server.flow_timeout_seconds,
        close_on_tcp_flags=False,
    ).extract(pcap_path)
    predictions: list[dict[str, Any]] = []
    input_view = InputView(server.view)
    for flow in flows:
        prefix_tokens = None
        if server.use_connection_tokens:
            prefix_tokens = [server.tokenizer.connection_type_token(flow.connection_type)]
        chunks = [
            PacketChunk(direction=packet.direction, data=packet.bytes_for_view(input_view))
            for packet in flow.packets
        ]
        windows = server.tokenizer.encode_flow(
            chunks,
            max_length=server.max_length,
            stride=server.stride,
            padding=True,
            prefix_tokens=prefix_tokens,
            byte_token_encoder=server.byte_token_encoder,
        )
        if server.max_windows is not None:
            windows = windows[: server.max_windows]
        input_ids = torch.tensor([[item.input_ids for item in windows]], dtype=torch.long)
        attention_mask = torch.tensor([[item.attention_mask for item in windows]], dtype=torch.long)
        window_mask = torch.ones(1, len(windows), dtype=torch.bool)
        with torch.no_grad():
            outputs = server.model(
                input_ids=input_ids.to(server.device_obj),
                attention_mask=attention_mask.to(server.device_obj),
                window_mask=window_mask.to(server.device_obj),
            )
        prediction = decode_hierarchical_prediction(
            outputs["major_logits"][0].cpu(),
            outputs["minor_logits"][0].cpu(),
            server.label_map,
            thresholds=server.thresholds,
        )
        item = prediction.as_dict()
        item.update(
            {
                "flow_id": flow.flow_id,
                "source_file": flow.source_file,
                "protocol": flow.protocol,
                "connection_type": flow.connection_type,
                "endpoint_a": flow.endpoint_a,
                "endpoint_b": flow.endpoint_b,
                "initiator_endpoint": flow.initiator_endpoint,
                "responder_endpoint": flow.responder_endpoint,
                "start_time": flow.start_time,
                "end_time": flow.end_time,
                "packet_count": flow.packet_count,
                "observed_packet_count": flow.observed_packet_count,
                "was_packet_truncated": flow.was_packet_truncated,
                "payload_byte_length": flow.payload_byte_length,
                "packet_byte_length": flow.packet_byte_length,
            }
        )
        predictions.append(item)
    return {
        "summary": _summarize_pcap_predictions(
            predictions,
            risk_threshold=server.risk_threshold,
            top_flows=server.top_flows,
            host_window_policies=server.host_window_policies,
        ),
        "flows": predictions[: server.max_returned_flows],
    }


class DemoHandler(BaseHTTPRequestHandler):
    server: "DemoServer"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        data = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0 or content_length > self.server.max_upload_bytes:
            self._send_json({"error": "invalid or too large upload"}, HTTPStatus.BAD_REQUEST)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type")},
        )
        file_item = form["pcap"] if "pcap" in form else None
        if file_item is None or not getattr(file_item, "filename", ""):
            self._send_json({"error": "missing pcap file"}, HTTPStatus.BAD_REQUEST)
            return
        filename = _safe_filename(file_item.filename)
        if not _is_allowed_upload(filename):
            self._send_json({"error": "only .pcap and .pcapng uploads are accepted"}, HTTPStatus.BAD_REQUEST)
            return

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="traffic_bert_demo_") as temp_dir:
            temp_root = Path(temp_dir)
            upload_path = temp_root / filename
            with open(upload_path, "wb") as handle:
                shutil.copyfileobj(file_item.file, handle)
            payload = _predict_pcap(self.server, upload_path)
            payload["elapsed_seconds"] = time.perf_counter() - started
            payload["uploaded_filename"] = filename
            self._send_json(payload)


class DemoServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        checkpoint: Path,
        cli_command: str,
        device: str,
        view: str,
        max_length: int,
        stride: int,
        max_windows: int,
        max_packets_per_flow: int | None,
        flow_timeout_seconds: float,
        thresholds: Path | None,
        semantic_codebook_path: Path | None,
        host_window_policy_path: Path | None,
        risk_threshold: float,
        top_flows: int,
        max_upload_mb: int,
        max_returned_flows: int,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.checkpoint = checkpoint
        self.cli_command = cli_command
        self.device = device
        self.view = view
        self.max_length = max_length
        self.stride = stride
        self.max_windows = max_windows
        self.max_packets_per_flow = max_packets_per_flow
        self.flow_timeout_seconds = flow_timeout_seconds
        self.risk_threshold = risk_threshold
        self.top_flows = top_flows
        self.host_window_policies = _load_host_window_policies(host_window_policy_path)
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self.max_returned_flows = max_returned_flows
        self.label_map = LabelMap.from_yaml(Path("configs/label_map.yaml"))
        self.semantic_codebook = _resolve_semantic_codebook(
            semantic_codebook_path,
            checkpoint=checkpoint,
        )
        self.tokenizer = ByteTokenizer(
            extra_tokens=self.semantic_codebook.tokens if self.semantic_codebook is not None else None
        )
        self.use_connection_tokens = _checkpoint_uses_connection_tokens(checkpoint)
        self.byte_token_encoder = None
        if self.semantic_codebook is not None:
            def encode_with_codebook(data: bytes) -> list[str]:
                return self.semantic_codebook.encode_bytes(data, tokenizer=self.tokenizer)

            self.byte_token_encoder = encode_with_codebook
        self.device_obj = resolve_device(device)
        self.model = _load_classifier(checkpoint, self.label_map, max_length)
        self.model.to(self.device_obj)
        self.model.eval()
        self.thresholds = _load_thresholds(thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local PCAP upload demo server.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--cli-command", default="traffic-bert")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--view", default="masked_header_packet")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--max-packets-per-flow", type=int, default=None)
    parser.add_argument("--flow-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--semantic-codebook-path", type=Path, default=None)
    parser.add_argument("--host-window-policy-path", type=Path, default=None)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--top-flows", type=int, default=20)
    parser.add_argument("--max-upload-mb", type=int, default=64)
    parser.add_argument("--max-returned-flows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    server = DemoServer(
        (args.host, args.port),
        DemoHandler,
        checkpoint=args.checkpoint,
        cli_command=args.cli_command,
        device=args.device,
        view=args.view,
        max_length=args.max_length,
        stride=args.stride,
        max_windows=args.max_windows,
        max_packets_per_flow=args.max_packets_per_flow,
        flow_timeout_seconds=args.flow_timeout_seconds,
        thresholds=args.thresholds,
        semantic_codebook_path=args.semantic_codebook_path,
        host_window_policy_path=args.host_window_policy_path,
        risk_threshold=args.risk_threshold,
        top_flows=args.top_flows,
        max_upload_mb=args.max_upload_mb,
        max_returned_flows=args.max_returned_flows,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"PCAP demo server listening on {url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
