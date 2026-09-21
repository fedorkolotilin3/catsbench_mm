#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MM_THREADS="${MM_NUM_THREADS:-$(nproc)}"
MM_DEVICE="${MM_DEVICE:-cpu}"
EXPERIMENT="${MM_EXPERIMENT:-dlight_sb_mm/benchmark_hd/d2_g002}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { echo "--device requires cpu or gpu" >&2; exit 2; }
      MM_DEVICE="$2"
      shift 2
      ;;
    --experiment)
      [[ $# -ge 2 ]] || { echo "--experiment requires a Hydra experiment name" >&2; exit 2; }
      EXPERIMENT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--device cpu|gpu] [--experiment HYDRA_EXPERIMENT]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${MM_DEVICE}" in
  cpu)
    LSE_BACKEND_VALUE=cpu
    ;;
  gpu)
    LSE_BACKEND_VALUE=any
    ;;
  *)
    echo "Unsupported MM device '${MM_DEVICE}'; expected cpu or gpu" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
export LSE_BACKEND="${LSE_BACKEND_VALUE}"
export CATS_MPLBACKEND="${CATS_MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${MM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${MM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${MM_THREADS}}"

echo "[mm] training ${EXPERIMENT} on ${MM_DEVICE} with ${MM_THREADS} CPU helper threads"
bash "${SCRIPT_DIR}/run_colab.sh" \
  "experiment=${EXPERIMENT}" \
  "trainer.accelerator=${MM_DEVICE}" \
  trainer.devices=1 \
  logger.csv.version=train

echo "[mm] testing last checkpoint"
bash "${SCRIPT_DIR}/run_colab.sh" \
  task_name=test \
  ckpt_path=auto \
  "experiment=${EXPERIMENT}" \
  "trainer.accelerator=${MM_DEVICE}" \
  trainer.devices=1 \
  logger.csv.version=test

echo "[mm] train and test completed"
