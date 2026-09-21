#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="${BASELINE_EXPERIMENT:-dlight_sb/benchmark_hd/d2_g002}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      [[ $# -ge 2 ]] || { echo "--experiment requires a Hydra experiment name" >&2; exit 2; }
      EXPERIMENT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--experiment HYDRA_EXPERIMENT]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${PROJECT_ROOT}"
export LSE_BACKEND="${LSE_BACKEND:-any}"
export CATS_MPLBACKEND="${CATS_MPLBACKEND:-Agg}"

echo "[baseline] training ${EXPERIMENT}"
bash "${SCRIPT_DIR}/run_colab.sh" \
  "experiment=${EXPERIMENT}" \
  logger=csv \
  logger.csv.version=train \
  '~callbacks.plotter_callback'

echo "[baseline] testing last checkpoint"
bash "${SCRIPT_DIR}/run_colab.sh" \
  task_name=test \
  ckpt_path=auto \
  "experiment=${EXPERIMENT}" \
  logger=csv \
  logger.csv.version=test \
  '~callbacks.plotter_callback'

echo "[baseline] train and test completed"
