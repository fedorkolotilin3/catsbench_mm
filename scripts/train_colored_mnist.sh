#!/bin/bash
#SBATCH --job-name=train-colored-mnist
#SBATCH --partition=ais-gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --array=2-5%2
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench

SEED=5
METHOD=alpha_csbm # (csbm, alpha_csbm, dlight_sb, dlight_sb_m)
EXPERIMENTS=(t2 t4 t10 t25 t50 t100)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  BASE=30000
  OFFSET=$(( SLURM_ARRAY_TASK_ID % 1000 ))
  export MASTER_PORT=$(( BASE + OFFSET ))
else
  export MASTER_PORT=29500
fi

python -m src.run \
  seed=${SEED} data.num_workers=4 data.pin_memory=true \
  experiment=${METHOD}/colored_mnist/${EXPERIMENT}
