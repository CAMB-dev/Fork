#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"

echo "== System =="
uname -a
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -a || true
fi

echo
echo "== ROCm tools =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E "Name:|Marketing Name|gfx" | head -n 40 || true
else
  echo "rocminfo: not found"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "rocm-smi: not found"
fi

echo
echo "== Python / PyTorch =="
if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv: not found" >&2
  exit 1
fi

"${UV_BIN}" run python - <<'PY'
import json
import sys

import torch

payload = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "hip": getattr(torch.version, "hip", None),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "devices": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else [],
}
print(json.dumps(payload, indent=2))
if not payload["cuda_available"]:
    raise SystemExit("PyTorch does not see a ROCm/CUDA device")
if payload["hip"] is None:
    raise SystemExit("torch.version.hip is None; this does not look like a ROCm torch build")
PY
