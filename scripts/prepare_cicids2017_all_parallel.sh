#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-.}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE="${FORCE:-0}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_PACKETS_PER_FLOW="${MAX_PACKETS_PER_FLOW:-32}"
MAX_PACKETS_TO_READ="${MAX_PACKETS_TO_READ:-}"
FLOW_TIMEOUT_SECONDS="${FLOW_TIMEOUT_SECONDS:-120}"
CLOSE_ON_TCP_FLAGS="${CLOSE_ON_TCP_FLAGS:-0}"
LOW_SUPPORT_MIN="${LOW_SUPPORT_MIN:-2}"
UV_BIN="${UV_BIN:-uv}"

cd "${WORKSPACE}"

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
  --dataset cicids2017
  --max-workers "${MAX_WORKERS}"
  --max-packets-per-flow "${MAX_PACKETS_PER_FLOW}"
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
if [[ "${FORCE}" == "1" ]]; then
  args+=(--force)
fi
if [[ "${SKIP_AUDIT:-0}" == "1" ]]; then
  args+=(--skip-audit)
fi
if [[ "${SKIP_VALIDATE:-0}" == "1" ]]; then
  args+=(--skip-validate)
fi

for item in \
  CICIDS_VIEWS=--cicids-views \
  WINDOW_SCOPE=--cicids-window-scope \
  CIC_LABEL_MAX_TIME_DELTA_SECONDS=--cicids-label-max-time-delta-seconds \
  MIN_ATTACK_FLOWS_PER_LABEL=--cicids-min-attack-flows-per-label \
  MIN_ATTACK_MATCH_RATIO=--cicids-min-attack-match-ratio \
  MAX_PER_MAJOR=--cicids-max-per-major \
  TIME_BLOCK_SIZE=--time-block-size; do
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
