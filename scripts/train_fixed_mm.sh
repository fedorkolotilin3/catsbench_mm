#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LSE_BACKEND=cpu
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m scripts.prepare_mm_experiment
python -m scripts.run_mm_experiment experiment=dlight_sb/fixed_cpu
python -m scripts.run_mm_experiment experiment=dlight_sb_mm/fixed_cpu
