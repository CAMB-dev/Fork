#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-32}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
MAX_PACKETS_TO_SKIP="${MAX_PACKETS_TO_SKIP:-0}"
FLOW_TIMEOUT_SECONDS="${FLOW_TIMEOUT_SECONDS:-120}"
CICIDS_LABEL_MAX_TIME_DELTA_SECONDS="${CICIDS_LABEL_MAX_TIME_DELTA_SECONDS:-900}"
NO_TIME_WINDOW="${NO_TIME_WINDOW:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cicids2017_alignment}"

cd "${WORKSPACE}"
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_cmd=("${PYTHON_BIN}")
elif [[ -x ".venv/bin/python" ]]; then
  python_cmd=(".venv/bin/python")
elif [[ -n "${UV_BIN:-}" ]]; then
  python_cmd=("${UV_BIN}" run python)
elif command -v uv >/dev/null 2>&1; then
  python_cmd=(uv run python)
else
  python_cmd=(python)
fi

jobs_json="${OUTPUT_DIR}/jobs.json"
cat >"${jobs_json}" <<'JSON'
[
  {
    "slug": "monday_benign",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Monday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_monday_benign.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_monday_benign.build.json"
  },
  {
    "slug": "tuesday_patator",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Tuesday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_tuesday_patator.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_tuesday_patator.build.json"
  },
  {
    "slug": "wednesday_dos",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Wednesday-workingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_wednesday_dos.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_wednesday_dos.build.json"
  },
  {
    "slug": "thursday_web",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Thursday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_thursday_web.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_thursday_web.build.json"
  },
  {
    "slug": "thursday_infiltration",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Thursday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_thursday_infiltration.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_thursday_infiltration.build.json"
  },
  {
    "slug": "friday_bot",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_friday_bot.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_friday_bot.build.json"
  },
  {
    "slug": "friday_portscan",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_friday_portscan.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_friday_portscan.build.json"
  },
  {
    "slug": "friday_ddos",
    "pcap_path": "data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap",
    "csv_path": "data/raw/CICIDS2017/labels/cicids2017_friday_ddos.csv",
    "build_json": "data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose/flows_friday_ddos.build.json"
  }
]
JSON

args=(
  scripts/diagnose_cicids_alignment.py
  --jobs-json "${jobs_json}"
  --output-dir "${OUTPUT_DIR}"
  --summary-output "${OUTPUT_DIR}/summary.json"
  --max-workers "${MAX_WORKERS}"
  --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
  --max-packets-to-skip "${MAX_PACKETS_TO_SKIP}"
  --flow-timeout-seconds "${FLOW_TIMEOUT_SECONDS}"
  --cic-label-max-time-delta-seconds "${CICIDS_LABEL_MAX_TIME_DELTA_SECONDS}"
)

if [[ -n "${MAX_PACKETS_TO_READ}" ]]; then
  args+=(--max-packets-to-read "${MAX_PACKETS_TO_READ}")
fi

if [[ "${NO_TIME_WINDOW}" == "1" ]]; then
  args+=(--no-time-window)
fi

echo "diagnose CICIDS2017 alignment -> ${OUTPUT_DIR}"
"${python_cmd[@]}" "${args[@]}"
