#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export LSE_BACKEND="${LSE_BACKEND:-any}"
export CATS_MPLBACKEND="${CATS_MPLBACKEND:-Agg}"

echo "[baseline] training dlight_sb/benchmark_hd/d2_g002"
bash "${SCRIPT_DIR}/run_colab.sh" \
  experiment=dlight_sb/benchmark_hd/d2_g002 \
  logger=csv \
  logger.csv.version=train \
  '~callbacks.plotter_callback'

echo "[baseline] testing last checkpoint"
bash "${SCRIPT_DIR}/run_colab.sh" \
  task_name=test \
  ckpt_path=auto \
  experiment=dlight_sb/benchmark_hd/d2_g002 \
  logger=csv \
  logger.csv.version=test \
  '~callbacks.plotter_callback'

echo "[baseline] train and test completed"
