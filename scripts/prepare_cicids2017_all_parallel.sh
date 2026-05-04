#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-16}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
PADDING_MINUTES="${PADDING_MINUTES:-20}"
MAX_PER_MAJOR="${MAX_PER_MAJOR:-50000}"
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

if [[ -f data/processed/cicids2017/all_split_label_stratified/train.parquet ]]; then
  "${UV_BIN}" run python scripts/audit_processed_split.py \
    --train-path data/processed/cicids2017/all_split_label_stratified/train.parquet \
    --val-path data/processed/cicids2017/all_split_label_stratified/val.parquet \
    --test-path data/processed/cicids2017/all_split_label_stratified/test.parquet \
    --output-path artifacts/cicids2017_all/split_audit.json
fi
