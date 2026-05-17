#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
DATASET="${DATASET:-cicids2017}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
RUN_SMOKE_TRAIN="${RUN_SMOKE_TRAIN:-0}"
MAX_WORKERS="${MAX_WORKERS:-${CICIDS_MAX_WORKERS:-5}}"
USTC_WORKERS="${USTC_WORKERS:-4}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-32}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
MAX_PACKETS_TO_SKIP="${MAX_PACKETS_TO_SKIP:-0}"
FLOW_TIMEOUT_SECONDS="${FLOW_TIMEOUT_SECONDS:-120}"
CLOSE_ON_TCP_FLAGS="${CLOSE_ON_TCP_FLAGS:-0}"
LOW_SUPPORT_MIN="${LOW_SUPPORT_MIN:-2}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

if [[ -n "${PROXY:-}" ]]; then
  export HTTP_PROXY="${PROXY}"
  export HTTPS_PROXY="${PROXY}"
fi
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
else
  export HF_ENDPOINT="https://hf-mirror.com"
fi

if command -v "${UV_BIN}" >/dev/null 2>&1; then
  PY_RUN=("${UV_BIN}" run python)
elif [[ -x ".venv/bin/python" ]]; then
  PY_RUN=(".venv/bin/python")
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY_RUN=(".venv/Scripts/python.exe")
else
  echo "Could not find ${UV_BIN} or project .venv Python." >&2
  exit 1
fi

args=(
  scripts/prepare_data.py
  --dataset "${DATASET}"
  --max-workers "${MAX_WORKERS}"
  --ustc-workers "${USTC_WORKERS}"
  --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
  --max-packets-to-skip "${MAX_PACKETS_TO_SKIP}"
  --flow-timeout-seconds "${FLOW_TIMEOUT_SECONDS}"
  "$([[ "${CLOSE_ON_TCP_FLAGS}" == "1" ]] && echo --close-on-tcp-flags || echo --no-close-on-tcp-flags)"
  --low-support-min "${LOW_SUPPORT_MIN}"
)

if [[ -n "${MAX_PACKETS_TO_READ}" ]]; then
  args+=(--max-packets-to-read "${MAX_PACKETS_TO_READ}")
fi
if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  args+=(--skip-download)
fi
if [[ "${FORCE_BUILD}" == "1" ]]; then
  args+=(--force)
fi
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  args+=(--skip-validate)
fi
if [[ "${SKIP_AUDIT}" == "1" ]]; then
  args+=(--skip-audit)
fi
if [[ "${RUN_SMOKE_TRAIN}" == "1" ]]; then
  args+=(--run-smoke-train)
fi

for item in \
  CICIDS_VIEWS=--cicids-views \
  CICIDS_WINDOW_SCOPE=--cicids-window-scope \
  CICIDS_LABEL_MAX_TIME_DELTA_SECONDS=--cicids-label-max-time-delta-seconds \
  CICIDS_MIN_ATTACK_FLOWS_PER_LABEL=--cicids-min-attack-flows-per-label \
  CICIDS_MIN_ATTACK_MATCH_RATIO=--cicids-min-attack-match-ratio \
  CICIDS_MAX_PER_MAJOR=--cicids-max-per-major \
  TIME_BLOCK_SIZE=--time-block-size \
  USTC_SOURCE_ROOT=--ustc-source-root; do
  name="${item%%=*}"
  flag="${item#*=}"
  value="${!name:-}"
  if [[ -n "${value}" ]]; then
    args+=("${flag}" "${value}")
  fi
done

if [[ "${CICIDS_DROP_EMPTY_PAYLOAD:-0}" == "1" ]]; then
  args+=(--cicids-drop-empty-payload)
fi

"${PY_RUN[@]}" "${args[@]}"
