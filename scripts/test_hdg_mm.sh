#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LSE_BACKEND=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Hydra resolves `auto` to the newest last.ckpt for this experiment and seed,
# and stores test metrics and artifacts in the same run directory.
python -m src.run \
  task_name=test \
  ckpt_path=auto \
  experiment=dlight_sb_mm/benchmark_hd/d2_g002 \
  logger.csv.version=test
