#!/bin/bash
#SBATCH --job-name=train-hgd
#SBATCH --partition=ais-gpu
#SBATCH --reservation=HPC-2966
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench

python scripts/train_hdg_local.py csbm --seed 5 --parallel 4