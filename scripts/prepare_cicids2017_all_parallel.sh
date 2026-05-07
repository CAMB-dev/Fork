#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-16}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
PADDING_MINUTES="${PADDING_MINUTES:-20}"
CSV_TIME_OFFSET_HOURS="${CSV_TIME_OFFSET_HOURS:-4}"
MAX_PER_MAJOR="${MAX_PER_MAJOR:-50000}"
MIN_ATTACK_FLOWS_PER_LABEL="${MIN_ATTACK_FLOWS_PER_LABEL:-1}"
TIME_BLOCK_SIZE="${TIME_BLOCK_SIZE:-512}"
LOW_SUPPORT_MIN="${LOW_SUPPORT_MIN:-2}"
NO_TIME_WINDOW="${NO_TIME_WINDOW:-0}"
FORCE="${FORCE:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  DATASET=cicids2017-all bash scripts/download_datasets.sh
fi

args=(
  run python scripts/prepare_cicids2017_all_parallel.py
  --max-workers "${MAX_WORKERS}"
  --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
  --padding-minutes "${PADDING_MINUTES}"
  --csv-time-offset-hours "${CSV_TIME_OFFSET_HOURS}"
  --max-per-major "${MAX_PER_MAJOR}"
  --min-attack-flows-per-label "${MIN_ATTACK_FLOWS_PER_LABEL}"
  --time-block-size "${TIME_BLOCK_SIZE}"
  --low-support-min "${LOW_SUPPORT_MIN}"
)

if [[ -n "${MAX_PACKETS_TO_READ}" ]]; then
  args+=(--max-packets-to-read "${MAX_PACKETS_TO_READ}")
fi
if [[ "${NO_TIME_WINDOW}" == "1" ]]; then
  args+=(--no-time-window)
fi
if [[ "${FORCE}" == "1" ]]; then
  args+=(--force)
fi

"${UV_BIN}" "${args[@]}"

audit_split() {
  local name="$1"
  local dir="$2"
  if [[ -f "${dir}/train.parquet" ]]; then
    "${UV_BIN}" run python scripts/audit_processed_split.py \
      --train-path "${dir}/train.parquet" \
      --val-path "${dir}/val.parquet" \
      --test-path "${dir}/test.parquet" \
      --low-support-min "${LOW_SUPPORT_MIN}" \
      --output-path "artifacts/cicids2017_all/${name}_split_audit.json"
  fi
}

audit_split random_flow data/processed/cicids2017/all_split_label_stratified
audit_split time_ordered data/processed/cicids2017/all_split_time_ordered
audit_split time_block data/processed/cicids2017/all_split_time_block
audit_split time_block_128 data/processed/cicids2017/all_split_time_block_128
