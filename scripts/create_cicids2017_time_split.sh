#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
INPUT_PATH="${INPUT_PATH:-data/processed/cicids2017/all_payload_only/flows_all.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/cicids2017/all_split_time_ordered}"
MAX_PER_MAJOR="${MAX_PER_MAJOR:-50000}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

"${UV_BIN}" run python scripts/create_time_ordered_split.py \
  --input-path "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --stratify-column major_label \
  --split-stratify-column source_label \
  --group-column flow_id \
  --time-column start_time \
  --max-per-class "${MAX_PER_MAJOR}"

"${UV_BIN}" run python scripts/audit_processed_split.py \
  --train-path "${OUTPUT_DIR}/train.parquet" \
  --val-path "${OUTPUT_DIR}/val.parquet" \
  --test-path "${OUTPUT_DIR}/test.parquet" \
  --output-path artifacts/cicids2017_time_ordered/split_audit.json
