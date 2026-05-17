#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
TRAIN_PATH="${TRAIN_PATH:-data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/train.parquet}"
VAL_PATH="${VAL_PATH:-data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/val.parquet}"
TEST_PATH="${TEST_PATH:-data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose/test.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cicids2017_masked_header_submode_group_formal_cuda_cap32_notcpclose_conn_b128_w4_sqrt_weighted}"
CICIDS_RAW_DIR="${CICIDS_RAW_DIR:-data/raw/CICIDS2017}"
CICIDS_PROCESSED_DIR="${CICIDS_PROCESSED_DIR:-data/processed/cicids2017/all_masked_header_packet_cap32_notcpclose}"
CICIDS_MERGED_PATH="${CICIDS_MERGED_PATH:-${CICIDS_PROCESSED_DIR}/flows_all.parquet}"
CICIDS_ATTACK_COVERAGE_PATH="${CICIDS_ATTACK_COVERAGE_PATH:-artifacts/cicids2017_coverage_masked_header_cap32_notcpclose/attack_coverage.json}"
CICIDS_REQUIRED_SPLIT_PARENT="${CICIDS_REQUIRED_SPLIT_PARENT:-data/processed/cicids2017/all_masked_header_split_submode_stratified_group_cap32_notcpclose}"
CICIDS_SPLIT_AUDIT_PATH="${CICIDS_SPLIT_AUDIT_PATH:-artifacts/cicids2017_all_masked_header_cap32_notcpclose/submode_stratified_group_split_audit.json}"
CICIDS_REQUIRED_VIEW="${CICIDS_REQUIRED_VIEW:-masked_header_packet}"
CICIDS_REQUIRED_KEEP_EMPTY_PAYLOAD="${CICIDS_REQUIRED_KEEP_EMPTY_PAYLOAD:-true}"
CICIDS_LABEL_MAX_TIME_DELTA_SECONDS="${CICIDS_LABEL_MAX_TIME_DELTA_SECONDS:-900}"
PLAN="${PLAN:-bert-supervised}"
VIEW="${VIEW:-masked_header_packet}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
STRIDE="${STRIDE:-384}"
MAX_WINDOWS="${MAX_WINDOWS:-2}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-3}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-32}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-3}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-128}"
CLASSIFIER_NUM_WORKERS="${CLASSIFIER_NUM_WORKERS:-4}"
CLASSIFIER_LEARNING_RATE="${CLASSIFIER_LEARNING_RATE:-3e-5}"
MAJOR_CLASS_WEIGHTING="${MAJOR_CLASS_WEIGHTING:-sqrt_balanced}"
MAJOR_CLASS_WEIGHT_CAP="${MAJOR_CLASS_WEIGHT_CAP:-20}"
USE_CONNECTION_TOKENS="${USE_CONNECTION_TOKENS:-1}"
USE_CONTEXT_TOKENS="${USE_CONTEXT_TOKENS:-0}"
USE_CONTEXT_FEATURES="${USE_CONTEXT_FEATURES:-0}"
SEMANTIC_CODEBOOK_PATH="${SEMANTIC_CODEBOOK_PATH:-}"
MLM_EPOCHS="${MLM_EPOCHS:-1}"
MLM_BATCH_SIZE="${MLM_BATCH_SIZE:-16}"
RESUME="${RESUME:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

if command -v "${UV_BIN}" >/dev/null 2>&1; then
  PY_RUN=("${UV_BIN}" run python)
  CLI_RUN=("${UV_BIN}" run traffic-bert)
elif [[ -x ".venv/bin/python" && -x ".venv/bin/traffic-bert" ]]; then
  PY_RUN=(".venv/bin/python")
  CLI_RUN=(".venv/bin/traffic-bert")
elif [[ -x ".venv/Scripts/python.exe" && -x ".venv/Scripts/traffic-bert.exe" ]]; then
  PY_RUN=(".venv/Scripts/python.exe")
  CLI_RUN=(".venv/Scripts/traffic-bert.exe")
else
  echo "Could not find ${UV_BIN} or project .venv traffic-bert entrypoints." >&2
  exit 1
fi

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
  "${PY_RUN[@]}" - <<'PY'
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

verify_formal_dataset() {
  local audit_path="${1:-}"
  local args=(
    scripts/verify_formal_dataset.py
    --dataset cicids2017
    --train-path "${TRAIN_PATH}"
    --val-path "${VAL_PATH}"
    --test-path "${TEST_PATH}"
    --raw-dir "${CICIDS_RAW_DIR}"
    --processed-dir "${CICIDS_PROCESSED_DIR}"
    --merged-path "${CICIDS_MERGED_PATH}"
    --attack-coverage-path "${CICIDS_ATTACK_COVERAGE_PATH}"
    --required-split-parent "${CICIDS_REQUIRED_SPLIT_PARENT}"
    --required-view "${CICIDS_REQUIRED_VIEW}"
    --required-keep-empty-payload "${CICIDS_REQUIRED_KEEP_EMPTY_PAYLOAD}"
    --required-window-scope pcap
    --required-csv-time-offset-hours 3.0
    --required-cic-label-max-time-delta-seconds "${CICIDS_LABEL_MAX_TIME_DELTA_SECONDS}"
    --output-path "${OUTPUT_ROOT}/formal_dataset_gate.json"
  )
  if [[ -n "${audit_path}" ]]; then
    args+=(--audit-path "${audit_path}")
  else
    args+=(--allow-missing-audit)
  fi
  "${PY_RUN[@]}" "${args[@]}"
}

require_path "${TRAIN_PATH}"
require_path "${VAL_PATH}"
require_path "${TEST_PATH}"
if [[ "${SKIP_AUDIT}" == "1" ]]; then
  echo "Formal CICIDS2017 training cannot skip split audit." >&2
  exit 1
fi
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
CLASSIFIER_NUM_WORKERS=${CLASSIFIER_NUM_WORKERS}
CLASSIFIER_LEARNING_RATE=${CLASSIFIER_LEARNING_RATE}
MAJOR_CLASS_WEIGHTING=${MAJOR_CLASS_WEIGHTING}
MAJOR_CLASS_WEIGHT_CAP=${MAJOR_CLASS_WEIGHT_CAP}
USE_CONNECTION_TOKENS=${USE_CONNECTION_TOKENS}
USE_CONTEXT_TOKENS=${USE_CONTEXT_TOKENS}
USE_CONTEXT_FEATURES=${USE_CONTEXT_FEATURES}
SEMANTIC_CODEBOOK_PATH=${SEMANTIC_CODEBOOK_PATH}
MLM_EPOCHS=${MLM_EPOCHS}
MLM_BATCH_SIZE=${MLM_BATCH_SIZE}
EOF

echo
echo "== Runtime =="
print_runtime
run_step "Verify CICIDS2017 formal dataset preflight" verify_formal_dataset

BASELINE_DIR="${OUTPUT_ROOT}/neural_baseline_cnn"
MLM_DIR="${OUTPUT_ROOT}/mlm"
CLASSIFIER_DIR="${OUTPUT_ROOT}/classifier"

audit_path="${OUTPUT_ROOT}/split_audit.json"
if [[ -f "${CICIDS_SPLIT_AUDIT_PATH}" ]]; then
  run_step "Reuse existing split audit" \
    cp "${CICIDS_SPLIT_AUDIT_PATH}" "${audit_path}"
else
  run_step "Audit processed split" \
    "${PY_RUN[@]}" scripts/audit_processed_split.py \
      --train-path "${TRAIN_PATH}" \
      --val-path "${VAL_PATH}" \
      --test-path "${TEST_PATH}" \
      --ignore-source-file-overlap-for-eligibility \
      --fail-on-warnings \
      --output-path "${audit_path}"
fi
run_step "Verify CICIDS2017 formal dataset audit gate" verify_formal_dataset "${audit_path}"

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
      "${CLI_RUN[@]}" train neural-baseline \
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
      "${CLI_RUN[@]}" eval neural-baseline \
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
      "${CLI_RUN[@]}" train mlm \
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

connection_token_args=()
if [[ "${USE_CONNECTION_TOKENS}" == "1" ]]; then
  connection_token_args=(--use-connection-tokens)
fi
context_token_args=()
if [[ "${USE_CONTEXT_TOKENS}" == "1" ]]; then
  context_token_args=(--use-context-tokens)
fi
context_feature_args=()
if [[ "${USE_CONTEXT_FEATURES}" == "1" ]]; then
  context_feature_args=(--use-context-features)
fi
semantic_codebook_args=()
if [[ -n "${SEMANTIC_CODEBOOK_PATH}" ]]; then
  require_path "${SEMANTIC_CODEBOOK_PATH}"
  semantic_codebook_args=(--semantic-codebook-path "${SEMANTIC_CODEBOOK_PATH}")
fi

if [[ "${do_classifier}" == "1" ]]; then
  classifier_checkpoint="${CLASSIFIER_DIR}/classifier.pt"
  if [[ "${RESUME}" == "1" && -f "${classifier_checkpoint}" ]]; then
    echo "Skipping classifier; checkpoint exists: ${classifier_checkpoint}"
  else
    run_step "Train Byte-BERT classifier" \
      "${CLI_RUN[@]}" train classifier \
        --train-path "${TRAIN_PATH}" \
        --val-path "${VAL_PATH}" \
        --output-dir "${CLASSIFIER_DIR}" \
        --view "${VIEW}" \
        --epochs "${CLASSIFIER_EPOCHS}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --num-workers "${CLASSIFIER_NUM_WORKERS}" \
        --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --major-class-weighting "${MAJOR_CLASS_WEIGHTING}" \
        --major-class-weight-cap "${MAJOR_CLASS_WEIGHT_CAP}" \
        "${connection_token_args[@]}" \
        "${context_token_args[@]}" \
        "${context_feature_args[@]}" \
        "${semantic_codebook_args[@]}" \
        "${init_bert_args[@]}"
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    classifier_eval_checkpoint="${CLASSIFIER_DIR}/classifier.best.pt"
    if [[ ! -f "${classifier_eval_checkpoint}" ]]; then
      classifier_eval_checkpoint="${classifier_checkpoint}"
    fi
    echo "Classifier eval checkpoint: ${classifier_eval_checkpoint}"

    threshold_path="${CLASSIFIER_DIR}/minor_thresholds.json"
    classifier_predictions_dir="${CLASSIFIER_DIR}/predictions"
    split_dir="$(dirname "${TRAIN_PATH}")"
    mkdir -p "${classifier_predictions_dir}"
    run_step "Calibrate classifier thresholds" \
      "${CLI_RUN[@]}" eval calibrate-thresholds \
        --data-path "${VAL_PATH}" \
        --checkpoint "${classifier_eval_checkpoint}" \
        --output-path "${threshold_path}" \
        --view "${VIEW}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --num-workers "${CLASSIFIER_NUM_WORKERS}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        "${connection_token_args[@]}" \
        "${context_token_args[@]}" \
        "${context_feature_args[@]}" \
        "${semantic_codebook_args[@]}"

    run_step "Evaluate Byte-BERT classifier on validation with predictions" \
      "${CLI_RUN[@]}" eval classifier \
        --data-path "${VAL_PATH}" \
        --checkpoint "${classifier_eval_checkpoint}" \
        --view "${VIEW}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --num-workers "${CLASSIFIER_NUM_WORKERS}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        "${connection_token_args[@]}" \
        "${context_token_args[@]}" \
        "${context_feature_args[@]}" \
        "${semantic_codebook_args[@]}" \
        --output-dir "${CLASSIFIER_DIR}/val_eval" \
        --output-predictions-path "${classifier_predictions_dir}/val_predictions.parquet"

    run_step "Evaluate Byte-BERT classifier on test with predictions" \
      "${CLI_RUN[@]}" eval classifier \
        --data-path "${TEST_PATH}" \
        --checkpoint "${classifier_eval_checkpoint}" \
        --view "${VIEW}" \
        --batch-size "${CLASSIFIER_BATCH_SIZE}" \
        --num-workers "${CLASSIFIER_NUM_WORKERS}" \
        --max-length "${MAX_LENGTH}" \
        --stride "${STRIDE}" \
        --max-windows "${MAX_WINDOWS}" \
        --device "${DEVICE}" \
        "${connection_token_args[@]}" \
        "${context_token_args[@]}" \
        "${context_feature_args[@]}" \
        "${semantic_codebook_args[@]}" \
        --output-dir "${CLASSIFIER_DIR}/test_eval" \
        --output-predictions-path "${classifier_predictions_dir}/test_predictions.parquet"

    run_step "Analyze Bot host-window risk layer" \
      "${PY_RUN[@]}" scripts/analyze_bot_host_windows.py \
        --split-dir "${split_dir}" \
        --prediction-dir "${classifier_predictions_dir}" \
        --output-path "${OUTPUT_ROOT}/bot_host_window_analysis.json" \
        --target-label botnet_malware \
        --window-seconds 30 60 300 \
        --min-recall 0.80 \
        --min-f1 0.75
  fi
fi

echo
run_step "Summarize formal run artifacts" \
  "${PY_RUN[@]}" scripts/summarize_formal_run.py \
    --output-root "${OUTPUT_ROOT}" \
    --write-json "${OUTPUT_ROOT}/formal_run_summary.json" \
    --fail-on-acceptance

echo
echo "Done. Outputs written to ${OUTPUT_ROOT}"
