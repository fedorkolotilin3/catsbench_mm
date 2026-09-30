#!/bin/bash
#SBATCH --job-name=train-celeba
#SBATCH --partition=ais-gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench

SEED=5
METHOD=alpha_csbm # (csbm, alpha_csbm, c2sbm)
EXPERIMENTS=(u0005 u001)
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  BASE=30000
  OFFSET=$(( SLURM_ARRAY_TASK_ID % 1000 ))
  export MASTER_PORT=$(( BASE + OFFSET ))
else
  export MASTER_PORT=29500
fi

python -m src.run \
  seed=${SEED} data.num_workers=2 data.pin_memory=true \
  experiment=${METHOD}/celeba/${EXPERIMENT}
