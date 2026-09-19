#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${CATS_VENV_DIR:-${PROJECT_ROOT}/.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Environment not found at ${VENV_DIR}. Run: bash scripts/setup_colab.sh" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export LSE_BACKEND="${LSE_BACKEND:-any}"
export PYTHONUNBUFFERED=1

exec "${VENV_DIR}/bin/python" -m src.run "$@"

