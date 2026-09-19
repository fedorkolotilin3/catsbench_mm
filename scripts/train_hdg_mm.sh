#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# DLightSBMM currently supports the exact CPU backend and float64 trainer.
export LSE_BACKEND=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

python -m src.run \
  experiment=dlight_sb_mm/benchmark_hd/d2_g002

