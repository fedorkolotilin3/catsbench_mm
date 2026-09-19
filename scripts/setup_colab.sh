#!/usr/bin/env bash
set -Eeuo pipefail

# Creates an isolated environment suitable for a Colab GPU runtime.
# Existing project files and configurations are not modified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${CATS_VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION="${CATS_PYTHON_VERSION:-3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m uv --version >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --upgrade uv
fi

# Colab can change its system Python between images (including Python 3.13
# builds where ensurepip is unavailable). uv downloads the requested Python and
# creates the environment without depending on the image's venv/ensurepip.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m uv venv --python "${PYTHON_VERSION}" --seed "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
echo "Using $("${VENV_PYTHON}" --version) at ${VENV_PYTHON}"

if [[ "${PIN_TORCH:-1}" == "1" ]]; then
  "${PYTHON_BIN}" -m uv pip install --python "${VENV_PYTHON}" \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
elif ! "${VENV_PYTHON}" -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "Installing current PyPI builds of PyTorch and torchvision."
  "${PYTHON_BIN}" -m uv pip install --python "${VENV_PYTHON}" torch torchvision
fi

"${PYTHON_BIN}" -m uv pip install --python "${VENV_PYTHON}" \
  -r "${PROJECT_ROOT}/requirements-colab.txt"
"${PYTHON_BIN}" -m uv pip install --python "${VENV_PYTHON}" \
  --no-deps --editable "${PROJECT_ROOT}"

# flash-attn is not needed for benchmark_hd and is expensive to build. It can
# still be installed explicitly for image experiments that require it.
if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m uv pip install --python "${VENV_PYTHON}" \
    flash-attn==2.7.3 --no-build-isolation
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
