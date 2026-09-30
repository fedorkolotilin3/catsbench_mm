#!/bin/bash
#SBATCH --job-name=train-hd-matching
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=6-00:00:00

source activate dot_bench
set -e

SEED=5
METHOD=alpha_csbm # (dlight_sb_m csbm alpha_csbm c2sbm)
DIMS=(2 16 64)
BENCHMARKS=(g002 g005 u0005 u001)
VARIANTS=(t63_kl t63_mse t15_kl t15_mse)

GROUP_ID=0
for DIM in "${DIMS[@]}"; do
  for BENCHMARK in "${BENCHMARKS[@]}"; do
    EXPERIMENTS=
    for VARIANT in "${VARIANTS[@]}"; do
      EXPERIMENTS+=${EXPERIMENTS:+,}${METHOD}/benchmark_hd/d${DIM}_${BENCHMARK}_${VARIANT}
    done
    RUN_ID=${SLURM_JOB_ID:?This script must be launched with sbatch}_${GROUP_ID}

    python -m src.run -m \
      hydra/launcher=submitit_local hydra.launcher.timeout_min=23040 hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 hydra.launcher.cpus_per_task=2 hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=0 data.pin_memory=false \
      'hydra.launcher.submitit_folder=${paths.log_dir}/.submitit/'${RUN_ID} \
      'hydra.sweep.subdir=${hydra:runtime.choices.experiment}/${seed}/${now:%Y-%m-%d}_${now:%H-%M-%S}'_${RUN_ID} \
      experiment=${EXPERIMENTS}

    GROUP_ID=$((GROUP_ID + 1))
  done
done
