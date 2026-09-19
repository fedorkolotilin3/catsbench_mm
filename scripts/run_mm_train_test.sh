#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MM_THREADS="${MM_NUM_THREADS:-$(nproc)}"

cd "${PROJECT_ROOT}"
export LSE_BACKEND=cpu
export CATS_MPLBACKEND="${CATS_MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${MM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${MM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${MM_THREADS}}"

echo "[mm] training dlight_sb_mm/benchmark_hd/d2_g002 with ${MM_THREADS} CPU threads"
bash "${SCRIPT_DIR}/run_colab.sh" \
  experiment=dlight_sb_mm/benchmark_hd/d2_g002 \
  logger.csv.version=train

echo "[mm] testing last checkpoint"
bash "${SCRIPT_DIR}/run_colab.sh" \
  task_name=test \
  ckpt_path=auto \
  experiment=dlight_sb_mm/benchmark_hd/d2_g002 \
  logger.csv.version=test

echo "[mm] train and test completed"
