#!/usr/bin/env bash
set -Eeuo pipefail

# Creates an isolated environment suitable for a Colab GPU runtime.
# Existing project files and configurations are not modified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${CATS_VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("catsbench requires Python 3.10 or newer")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  # Colab supplies a CUDA-compatible PyTorch build. system-site-packages lets the
  # venv reuse it instead of downloading another multi-gigabyte wheel.
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ "${PIN_TORCH:-0}" == "1" ]]; then
  "${VENV_PYTHON}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
elif ! "${VENV_PYTHON}" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "PyTorch is not available in this runtime; installing current PyPI builds."
  "${VENV_PYTHON}" -m pip install torch torchvision
fi

"${VENV_PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements-colab.txt"
"${VENV_PYTHON}" -m pip install --no-deps --editable "${PROJECT_ROOT}"

# flash-attn is not needed for benchmark_hd and is expensive to build. It can
# still be installed explicitly for image experiments that require it.
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  "${VENV_PYTHON}" -m pip install flash-attn==2.7.3 --no-build-isolation
fi

"${VENV_PYTHON}" - <<'PY'
import hydra
import lightning
import ot
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is unavailable. In Colab select a GPU runtime and rerun setup.")
print("Colab environment is ready.")
PY

echo
echo "Run an experiment with:"
echo "  bash scripts/run_colab.sh experiment=dlight_sb/benchmark_hd/d2_g002 logger=csv logger.csv.version=train '~callbacks.plotter_callback'"

