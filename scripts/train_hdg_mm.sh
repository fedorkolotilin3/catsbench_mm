#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Baseline, MM, prior and benchmark share the custom CUDA kernel.
export LSE_BACKEND=triton
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

python -m src.run \
  experiment=dlight_sb_mm/benchmark_hd/d2_g002 \
  logger.csv.version=train
