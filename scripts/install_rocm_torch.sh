#!/usr/bin/env bash
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-6.4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm${ROCM_VERSION}}"

echo "== ROCm PyTorch install =="
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "UV_BIN=${UV_BIN}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv was not found in PATH. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

"${UV_BIN}" venv --python "${PYTHON_BIN}"
"${UV_BIN}" pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
"${UV_BIN}" pip install -e .

"${UV_BIN}" run "${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("hip:", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu_0:", torch.cuda.get_device_name(0))
PY
