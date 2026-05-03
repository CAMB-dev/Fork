#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
DATASET="${DATASET:-all}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_REPO_ID="${HF_REPO_ID:-bencorn/CICIDS2017}"
CICIDS_RAW_DIR="${CICIDS_RAW_DIR:-data/raw/CICIDS2017}"
USTC_RAW_DIR="${USTC_RAW_DIR:-data/raw/USTC-TFC2016}"
USTC_SOURCE_URL="${USTC_SOURCE_URL:-https://github.com/yungshenglu/USTC-TFC2016/archive/refs/heads/master.zip}"
USTC_ARCHIVE_PATH="${USTC_ARCHIVE_PATH:-${USTC_RAW_DIR}/USTC-TFC2016-master.zip}"
USTC_EXTRACT_DIR="${USTC_EXTRACT_DIR:-${USTC_RAW_DIR}/extracted}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"
export HF_ENDPOINT
export HF_REPO_ID
export CICIDS_RAW_DIR
export USTC_ARCHIVE_PATH
export USTC_EXTRACT_DIR

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  if [[ -f "${output}" ]]; then
    echo "skip existing: ${output}"
    return 0
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "$(basename "${output}")" -d "$(dirname "${output}")" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "Need aria2c, curl, or wget for URL downloads" >&2
    exit 1
  fi
}

extract_7z_archives() {
  local root="$1"
  local seven_zip=""
  if command -v 7zz >/dev/null 2>&1; then
    seven_zip="7zz"
  elif command -v 7z >/dev/null 2>&1; then
    seven_zip="7z"
  else
    echo "7z/7zz not found; nested USTC .7z archives were not extracted." >&2
    echo "Install p7zip-full or 7zip, then rerun DATASET=ustc_tfc2016 bash scripts/download_datasets.sh." >&2
    return 0
  fi

  while IFS= read -r -d "" archive; do
    local target="${archive%.7z}"
    if [[ -d "${target}" && -n "$(find "${target}" -type f -name '*.pcap' -print -quit)" ]]; then
      echo "skip existing 7z extract: ${target}"
      continue
    fi
    mkdir -p "${target}"
    echo "extract: ${archive}"
    "${seven_zip}" x -y "-o${target}" "${archive}" >/dev/null
  done < <(find "${root}" -name "*.7z" -type f -print0 | sort -z)
}

download_cicids_friday() {
  echo "== Download CICIDS2017 Friday via HuggingFace =="
  echo "HF_ENDPOINT=${HF_ENDPOINT}"
  "${UV_BIN}" run python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ.get("HF_REPO_ID", "bencorn/CICIDS2017")
raw_dir = Path(os.environ.get("CICIDS_RAW_DIR", "data/raw/CICIDS2017"))
raw_dir.mkdir(parents=True, exist_ok=True)
for filename in ["pcaps/Friday-WorkingHours.pcap", "csvs/GeneratedLabelledFlows.zip"]:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=raw_dir,
    )
    print(path)
PY
}

download_ustc() {
  echo "== Download USTC-TFC2016 =="
  echo "USTC_SOURCE_URL=${USTC_SOURCE_URL}"
  download_file "${USTC_SOURCE_URL}" "${USTC_ARCHIVE_PATH}"
  mkdir -p "${USTC_EXTRACT_DIR}"
  if [[ -d "${USTC_EXTRACT_DIR}/USTC-TFC2016-master" ]]; then
    echo "skip existing extract: ${USTC_EXTRACT_DIR}/USTC-TFC2016-master"
    return 0
  fi
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "${USTC_ARCHIVE_PATH}" -d "${USTC_EXTRACT_DIR}"
  else
    "${UV_BIN}" run python - <<'PY'
import os
import zipfile

archive = os.environ["USTC_ARCHIVE_PATH"]
extract_dir = os.environ["USTC_EXTRACT_DIR"]
with zipfile.ZipFile(archive) as handle:
    handle.extractall(extract_dir)
PY
  fi
  extract_7z_archives "${USTC_EXTRACT_DIR}/USTC-TFC2016-master"
}

case "${DATASET}" in
  cicids2017-friday-smoke|cicids2017)
    download_cicids_friday
    ;;
  ustc|ustc_tfc2016)
    download_ustc
    ;;
  all)
    download_cicids_friday
    download_ustc
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}. Use cicids2017-friday-smoke, cicids2017, ustc_tfc2016, ustc, or all." >&2
    exit 1
    ;;
esac
