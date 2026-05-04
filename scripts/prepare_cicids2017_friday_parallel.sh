#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
MAX_WORKERS="${MAX_WORKERS:-3}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-16}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
PADDING_MINUTES="${PADDING_MINUTES:-20}"
MAX_PER_MAJOR="${MAX_PER_MAJOR:-50000}"
NO_TIME_WINDOW="${NO_TIME_WINDOW:-0}"
FORCE="${FORCE:-0}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

args=(
  run python scripts/prepare_cicids2017_friday_parallel.py
  --max-workers "${MAX_WORKERS}"
  --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
  --padding-minutes "${PADDING_MINUTES}"
  --max-per-major "${MAX_PER_MAJOR}"
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

if [[ -f data/processed/cicids2017/friday_split_label_stratified/train.parquet ]]; then
  "${UV_BIN}" run python scripts/audit_processed_split.py \
    --train-path data/processed/cicids2017/friday_split_label_stratified/train.parquet \
    --val-path data/processed/cicids2017/friday_split_label_stratified/val.parquet \
    --test-path data/processed/cicids2017/friday_split_label_stratified/test.parquet \
    --output-path artifacts/cicids2017_friday/split_audit.json
fi
