#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WINDOW_SCOPE="${WINDOW_SCOPE:-pcap}"
CIC_LABEL_MAX_TIME_DELTA_SECONDS="${CIC_LABEL_MAX_TIME_DELTA_SECONDS:-900}"
PLAN="${PLAN:-bert-supervised}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cicids2017_masked_header_submode_group_formal_cuda}"
CICIDS_VIEWS="${CICIDS_VIEWS:-masked_header_packet}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-128}"
CLASSIFIER_NUM_WORKERS="${CLASSIFIER_NUM_WORKERS:-4}"
MAJOR_CLASS_WEIGHTING="${MAJOR_CLASS_WEIGHTING:-none}"

cd "${WORKSPACE}"
export HF_ENDPOINT
export WINDOW_SCOPE
export CIC_LABEL_MAX_TIME_DELTA_SECONDS
export PLAN
export OUTPUT_ROOT
export CICIDS_VIEWS
export CLASSIFIER_BATCH_SIZE
export CLASSIFIER_NUM_WORKERS
export MAJOR_CLASS_WEIGHTING

run_step() {
  local name="$1"
  shift
  echo
  echo "==== ${name} ===="
  "$@"
}

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  run_step "Download CICIDS2017 full PCAPs and labels" \
    env DATASET=cicids2017-all bash scripts/download_datasets.sh
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  run_step "Prepare CICIDS2017 formal full-PCAP dataset" \
    env FORCE="${FORCE_PREPARE}" bash scripts/prepare_cicids2017_all_parallel.sh
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  run_step "Train and evaluate CICIDS2017 formal submode-stratified group run" \
    bash scripts/train_cicids_formal_cuda.sh
fi

echo
echo "Cloud formal run finished."
echo "Output root: ${OUTPUT_ROOT}"
echo "Summary: ${OUTPUT_ROOT}/formal_run_summary.json"
