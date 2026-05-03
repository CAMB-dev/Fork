#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
DATASET="${DATASET:-cicids2017-friday-smoke}"
PROXY="${PROXY:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
RUN_SMOKE_TRAIN="${RUN_SMOKE_TRAIN:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-250000}"
MAX_PACKETS_TO_SKIP="${MAX_PACKETS_TO_SKIP:-0}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-16}"
MAX_PER_MAJOR="${MAX_PER_MAJOR:-2000}"
LABEL_FILE_CONTAINS="${LABEL_FILE_CONTAINS:-}"
START_TIME="${START_TIME:-}"
END_TIME="${END_TIME:-}"
USTC_SOURCE_ROOT="${USTC_SOURCE_ROOT:-data/raw/USTC-TFC2016/extracted/USTC-TFC2016-master}"
USTC_MERGED_PATH="${USTC_MERGED_PATH:-data/processed/ustc_tfc2016/merged/payload_only.parquet}"
USTC_LABEL_SPLIT_DIR="${USTC_LABEL_SPLIT_DIR:-data/processed/ustc_tfc2016/split_label_stratified}"
USTC_SOURCE_SPLIT_DIR="${USTC_SOURCE_SPLIT_DIR:-data/processed/ustc_tfc2016/split_source_file}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"
export HF_ENDPOINT

run_step() {
  local name="$1"
  shift
  echo
  echo "==== ${name} ===="
  "$@"
}

validate_split() {
  local split_dir="$1"
  for split in train val test; do
    local path="${split_dir}/${split}.parquet"
    if [[ -f "${path}" ]]; then
      run_step "Validate ${path}" "${UV_BIN}" run traffic-bert data validate --input-path "${path}"
    fi
  done
}

audit_split() {
  local split_dir="$1"
  local output_path="$2"
  local train="${split_dir}/train.parquet"
  local val="${split_dir}/val.parquet"
  local test="${split_dir}/test.parquet"
  if [[ -f "${train}" && -f "${val}" && -f "${test}" ]]; then
    run_step "Audit ${split_dir}" \
      "${UV_BIN}" run python scripts/audit_processed_split.py \
        --train-path "${train}" \
        --val-path "${val}" \
        --test-path "${test}" \
        --output-path "${output_path}"
  fi
}

prepare_cicids_friday_smoke() {
  if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    run_step "Download CICIDS2017 Friday smoke files" \
      env DATASET=cicids2017-friday-smoke bash scripts/download_datasets.sh
  fi
  local args=(
    run python scripts/cicids2017_friday_smoke.py
    --max-packets-to-read "${MAX_PACKETS_TO_READ}"
    --max-packets-to-skip "${MAX_PACKETS_TO_SKIP}"
    --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
    --max-per-major "${MAX_PER_MAJOR}"
  )
  if [[ -n "${START_TIME}" ]]; then
    args+=(--start-time "${START_TIME}")
  fi
  if [[ -n "${END_TIME}" ]]; then
    args+=(--end-time "${END_TIME}")
  fi
  if [[ -n "${LABEL_FILE_CONTAINS}" ]]; then
    args+=(--label-file-contains "${LABEL_FILE_CONTAINS}")
  fi
  if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    args+=(--skip-download)
  fi
  if [[ "${FORCE_BUILD}" == "1" ]]; then
    args+=(--force-build)
  fi
  if [[ "${RUN_SMOKE_TRAIN}" == "1" ]]; then
    args+=(--run-train)
  fi

  run_step "Prepare CICIDS2017 Friday smoke" "${UV_BIN}" "${args[@]}"

  if [[ "${SKIP_VALIDATE}" != "1" ]]; then
    validate_split "data/processed/cicids2017/smoke"
  fi
  if [[ "${SKIP_AUDIT}" != "1" ]]; then
    audit_split "data/processed/cicids2017/smoke" "artifacts/cicids2017_smoke/split_audit.json"
  fi
}

prepare_ustc() {
  if [[ "${SKIP_DOWNLOAD}" != "1" && ! -d "${USTC_SOURCE_ROOT}" ]]; then
    run_step "Download USTC-TFC2016" env DATASET=ustc bash scripts/download_datasets.sh
  fi
  if [[ ! -d "${USTC_SOURCE_ROOT}" ]]; then
    echo "USTC source root not found: ${USTC_SOURCE_ROOT}" >&2
    echo "Place/extract USTC-TFC2016 there or set USTC_SOURCE_ROOT. Skipping USTC." >&2
    return 0
  fi

  mkdir -p data/processed/ustc_tfc2016/files data/processed/ustc_tfc2016/logs
  run_step "Preprocess USTC-TFC2016 PCAP files" bash -c '
    set -euo pipefail
    source_root="$1"
    max_packets_per_flow="$2"
    while IFS= read -r -d "" pcap; do
      if [[ "${pcap}" == *"/Benign/"* ]]; then
        kind="Benign"
        label_args=(--label-source static --static-label BENIGN)
      else
        kind="Malware"
        label_args=(--label-source filename)
      fi
      base="$(basename "${pcap}" .pcap)"
      safe_base="$(printf "%s" "${base}" | tr " /:*?\"<>|" "_")"
      output="data/processed/ustc_tfc2016/files/${kind}_${safe_base}.parquet"
      stdout="data/processed/ustc_tfc2016/logs/${kind}_${safe_base}.stdout.log"
      stderr="data/processed/ustc_tfc2016/logs/${kind}_${safe_base}.stderr.log"
      if [[ -f "${output}" ]]; then
        echo "skip existing: ${output}"
        continue
      fi
      echo "build: ${pcap}"
      uv run traffic-bert data build \
        --input-path "${pcap}" \
        --output-path "${output}" \
        --source-dataset ustc-tfc2016 \
        --split train \
        --views payload_only \
        --max-packets-per-flow "${max_packets_per_flow}" \
        --no-keep-empty-payload \
        "${label_args[@]}" \
        >"${stdout}" 2>"${stderr}"
    done < <(find "${source_root}" -name "*.pcap" -type f -print0 | sort -z)
  ' bash "${USTC_SOURCE_ROOT}" "${MAX_PACKETS_PER_FLOW}"

  mapfile -t files < <(find data/processed/ustc_tfc2016/files -name '*.parquet' -type f | sort)
  if [[ "${#files[@]}" == "0" ]]; then
    echo "No USTC processed parquet files found under data/processed/ustc_tfc2016/files" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${USTC_MERGED_PATH}")"
  local merge_args=(run traffic-bert data merge)
  for file in "${files[@]}"; do
    merge_args+=(--input-path "${file}")
  done
  merge_args+=(--output-path "${USTC_MERGED_PATH}")
  run_step "Merge USTC processed files" "${UV_BIN}" "${merge_args[@]}"

  run_step "Create USTC label-stratified split" \
    "${UV_BIN}" run traffic-bert data sample-stratified \
      --input-path "${USTC_MERGED_PATH}" \
      --output-dir "${USTC_LABEL_SPLIT_DIR}" \
      --stratify-column major_label \
      --split-stratify-column source_label \
      --group-column flow_id \
      --max-per-class 1000000000

  run_step "Create USTC source-file split" \
    "${UV_BIN}" run traffic-bert data split \
      --input-path "${USTC_MERGED_PATH}" \
      --output-dir "${USTC_SOURCE_SPLIT_DIR}" \
      --group-column source_file

  if [[ "${SKIP_VALIDATE}" != "1" ]]; then
    validate_split "${USTC_LABEL_SPLIT_DIR}"
    validate_split "${USTC_SOURCE_SPLIT_DIR}"
  fi
  if [[ "${SKIP_AUDIT}" != "1" ]]; then
    audit_split "${USTC_LABEL_SPLIT_DIR}" "artifacts/ustc_tfc2016/split_label_stratified_audit.json"
    audit_split "${USTC_SOURCE_SPLIT_DIR}" "artifacts/ustc_tfc2016/split_source_file_audit.json"
  fi
}

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH" >&2
  exit 1
fi

if [[ -n "${PROXY}" ]]; then
  export HTTP_PROXY="${PROXY}"
  export HTTPS_PROXY="${PROXY}"
fi

case "${DATASET}" in
  cicids2017-friday-smoke)
    prepare_cicids_friday_smoke
    ;;
  ustc)
    prepare_ustc
    ;;
  all)
    prepare_cicids_friday_smoke
    prepare_ustc
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}. Use cicids2017-friday-smoke, ustc, or all." >&2
    exit 1
    ;;
esac
