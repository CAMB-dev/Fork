#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-16}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
NO_TIME_WINDOW="${NO_TIME_WINDOW:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cicids2017_alignment}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"
mkdir -p "${OUTPUT_DIR}"

run_diag() {
  local slug="$1"
  local pcap="$2"
  local csv="$3"
  local build_json="$4"
  local output="${OUTPUT_DIR}/${slug}.json"

  local args=(
    run python scripts/diagnose_cicids_alignment.py
    --pcap-path "${pcap}"
    --csv-path "${csv}"
    --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
  )

  if [[ -n "${MAX_PACKETS_TO_READ}" ]]; then
    args+=(--max-packets-to-read "${MAX_PACKETS_TO_READ}")
  fi

  if [[ "${NO_TIME_WINDOW}" != "1" && -f "${build_json}" ]]; then
    local min_time
    local max_time
    min_time="$("${UV_BIN}" run python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("min_packet_time") or "")' "${build_json}")"
    max_time="$("${UV_BIN}" run python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("max_packet_time") or "")' "${build_json}")"
    if [[ -n "${min_time}" ]]; then
      args+=(--min-packet-time "${min_time}")
    fi
    if [[ -n "${max_time}" ]]; then
      args+=(--max-packet-time "${max_time}")
    fi
  fi

  echo "diagnose ${slug} -> ${output}"
  "${UV_BIN}" "${args[@]}" >"${output}"
}

start_job() {
  run_diag "$@" &
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_WORKERS}" ]]; do
    wait -n
  done
}

start_job \
  monday_benign \
  data/raw/CICIDS2017/pcaps/Monday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_monday_benign.csv \
  data/processed/cicids2017/all_payload_only/flows_monday_benign.build.json

start_job \
  tuesday_patator \
  data/raw/CICIDS2017/pcaps/Tuesday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_tuesday_patator.csv \
  data/processed/cicids2017/all_payload_only/flows_tuesday_patator.build.json

start_job \
  wednesday_dos \
  data/raw/CICIDS2017/pcaps/Wednesday-workingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_wednesday_dos.csv \
  data/processed/cicids2017/all_payload_only/flows_wednesday_dos.build.json

start_job \
  thursday_web \
  data/raw/CICIDS2017/pcaps/Thursday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_thursday_web.csv \
  data/processed/cicids2017/all_payload_only/flows_thursday_web.build.json

start_job \
  thursday_infiltration \
  data/raw/CICIDS2017/pcaps/Thursday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_thursday_infiltration.csv \
  data/processed/cicids2017/all_payload_only/flows_thursday_infiltration.build.json

start_job \
  friday_bot \
  data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_friday_bot.csv \
  data/processed/cicids2017/all_payload_only/flows_friday_bot.build.json

start_job \
  friday_portscan \
  data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_friday_portscan.csv \
  data/processed/cicids2017/all_payload_only/flows_friday_portscan.build.json

start_job \
  friday_ddos \
  data/raw/CICIDS2017/pcaps/Friday-WorkingHours.pcap \
  data/raw/CICIDS2017/labels/cicids2017_friday_ddos.csv \
  data/processed/cicids2017/all_payload_only/flows_friday_ddos.build.json

wait

"${UV_BIN}" run python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/cicids2017_alignment")
summary = {}
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary[path.stem] = {
        "csv_label_counts": payload.get("csv_label_counts", {}),
        "pcap_flows": payload.get("pcap_flows"),
        "key_intersection": payload.get("key_intersection"),
        "attack_key_intersection": payload.get("attack_key_intersection"),
        "matched_labels_from_pcap_flows": payload.get("matched_labels_from_pcap_flows", {}),
        "payload_matched_labels": payload.get("payload_matched_labels", {}),
        "unmatched_pcap_flows": payload.get("unmatched_pcap_flows"),
    }
out = root / "summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
