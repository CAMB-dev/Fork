#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
TRAIN_PATH="${TRAIN_PATH:-data/processed/ustc_tfc2016/split_source_file/train.parquet}"
VAL_PATH="${VAL_PATH:-data/processed/ustc_tfc2016/split_source_file/val.parquet}"
TEST_PATH="${TEST_PATH:-data/processed/ustc_tfc2016/split_source_file/test.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/ustc_formal_cuda}"
PLAN="${PLAN:-bert-supervised}"
VIEW="${VIEW:-payload_only}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
STRIDE="${STRIDE:-384}"
MAX_WINDOWS="${MAX_WINDOWS:-2}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-3}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-32}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-3}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-16}"
MLM_EPOCHS="${MLM_EPOCHS:-1}"
MLM_BATCH_SIZE="${MLM_BATCH_SIZE:-16}"
RESUME="${RESUME:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path not found: ${path}" >&2
    exit 1
  fi
}

run_step() {
  local name="$1"
  shift
  echo
  echo "==== ${name} ===="
  "$@"
}

print_runtime() {
  "${UV_BIN}" run python - <<'PY'
import json
import torch

payload = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda": getattr(torch.version, "cuda", None),
    "hip": getattr(torch.version, "hip", None),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "devices": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": torch.cuda.get_device_capability(index),
            "total_memory_gb": round(
                torch.cuda.get_device_properties(index).total_memory / (1024**3), 2
            ),
        }
        for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else [],
}
print(json.dumps(payload, indent=2))
PY
}

require_path "${TRAIN_PATH}"
require_path "${VAL_PATH}"
require_path "${TEST_PATH}"
mkdir -p "${OUTPUT_ROOT}"

echo "== Training config =="
cat <<EOF
PLAN=${PLAN}
TRAIN_PATH=${TRAIN_PATH}
VAL_PATH=${VAL_PATH}
TEST_PATH=${TEST_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
VIEW=${VIEW}
DEVICE=${DEVICE}
MAX_LENGTH=${MAX_LENGTH}
STRIDE=${STRIDE}
MAX_WINDOWS=${MAX_WINDOWS}
CLASSIFIER_EPOCHS=${CLASSIFIER_EPOCHS}
CLASSIFIER_BATCH_SIZE=${CLASSIFIER_BATCH_SIZE}
MLM_EPOCHS=${MLM_EPOCHS}
MLM_BATCH_SIZE=${MLM_BATCH_SIZE}
EOF

echo
echo "== Runtime =="
print_runtime

BASELINE_DIR="${OUTPUT_ROOT}/neural_baseline_cnn"
MLM_DIR="${OUTPUT_ROOT}/mlm"
CLASSIFIER_DIR="${OUTPUT_ROOT}/classifier"

if [[ "${SKIP_AUDIT}" != "1" ]]; then
  run_step "Audit processed split" \
    "${UV_BIN}" run python scripts/audit_processed_split.py \
      --train-path "${TRAIN_PATH}" \
      --val-path "${VAL_PATH}" \
      --test-path "${TEST_PATH}" \
      --output-path "${OUTPUT_ROOT}/split_audit.json"
fi

do_baseline=0
do_mlm=0
do_classifier=0
case "${PLAN}" in
  baseline)
    do_baseline=1
    ;;
  bert-supervised)
    do_classifier=1
    ;;
  bert-mlm)
    do_mlm=1
    do_classifier=1
    ;;
  full)
    do_baseline=1
    do_mlm=1
    do_classifier=1
    ;;
  *)
    echo "Unsupported PLAN=${PLAN}. Use baseline, bert-supervised, bert-mlm, or full." >&2
    exit 1
    ;;
esac

if [[ "${do_baseline}" == "1" ]]; then
  baseline_checkpoint="${BASELINE_DIR}/neural_baseline.pt"
  if [[ "${RESUME}" == "1" && -f "${baseline_checkpoint}" ]]; then
    echo "Skipping CNN baseline; checkpoint exists: ${baseline_checkpoint}"
  else
    run_step "Train CNN baseline" \
      "${UV_BIN}" run traffic-bert train neural-baseline \
        --train-path "${TRAIN_PATH}" \
        --val-path "${VAL_PATH}" \
        --output-dir "${BASELINE_DIR}" \
        --view "${VIEW}" \
        --model cnn \
        --epochs "${BASELINE_EPOCHS}" \
        --batch-size "${BASELINE_BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    run_step "Evaluate CNN baseline" \
      "${UV_BIN}" run traffic-bert eval neural-baseline \
        --data-path "${TEST_PATH}" \
        --checkpoint "${baseline_checkpoint}" \
        --view "${VIEW}" \
        --batch-size "${BASELINE_BATCH_SIZE}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --output-dir "${BASELINE_DIR}/test_eval"
  fi
fi

init_bert_args=()
if [[ "${do_mlm}" == "1" ]]; then
  mlm_checkpoint="${MLM_DIR}/mlm.pt"
  if [[ "${RESUME}" == "1" && -f "${mlm_checkpoint}" ]]; then
    echo "Skipping MLM; checkpoint exists: ${mlm_checkpoint}"
  else
    run_step "Train MLM" \
      "${UV_BIN}" run traffic-bert train mlm \
        --train-path "${TRAIN_PATH}" \
        --output-dir "${MLM_DIR}" \
        --view "${VIEW}" \
        --epochs "${MLM_EPOCHS}" \
        --batch-size "${MLM_BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
  fi
  init_bert_args=(--init-bert-checkpoint "${mlm_checkpoint}")
fi

if [[ "${do_classifier}" == "1" ]]; then
  classifier_checkpoint="${CLASSIFIER_DIR}/classifier.pt"
  if [[ "${RESUME}" == "1" && -f "${classifier_checkpoint}" ]]; then
    echo "Skipping classifier; checkpoint exists: ${classifier_checkpoint}"
  else
    run_step "Train Byte-BERT classifier" \
      "${UV_BIN}" run traffic-bert train classifier \
        --train-path "${TRAIN_PATH}" \
        --val-path "${VAL_PATH}" \
        --output-dir "${CLASSIFIER_DIR}" \
        --view "${VIEW}" \
        --epochs "${CLASSIFIER_EPOCHS}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        "${init_bert_args[@]}"
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    threshold_path="${CLASSIFIER_DIR}/minor_thresholds.json"
    run_step "Calibrate classifier thresholds" \
      "${UV_BIN}" run traffic-bert eval calibrate-thresholds \
        --data-path "${VAL_PATH}" \
        --checkpoint "${classifier_checkpoint}" \
        --output-path "${threshold_path}" \
        --view "${VIEW}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}"

    run_step "Evaluate Byte-BERT classifier" \
      "${UV_BIN}" run traffic-bert eval classifier \
        --data-path "${TEST_PATH}" \
        --checkpoint "${classifier_checkpoint}" \
        --view "${VIEW}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --output-dir "${CLASSIFIER_DIR}/test_eval"
  fi
fi

echo
echo "Done. Outputs written to ${OUTPUT_ROOT}"
