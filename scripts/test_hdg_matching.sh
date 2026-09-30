#!/bin/bash
#SBATCH --job-name=test-hd-matching
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --array=0-47%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00

sleep $((SLURM_ARRAY_TASK_ID * 5))
source activate dot_bench
SEED=2
METHOD=csbm # (dlight_sb_m csbm alpha_csbm c2sbm)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g002_t63_kl task_name=test ckpt_path=auto
    ;;
  1)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g002_t63_mse task_name=test ckpt_path=auto
    ;;
  2)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g002_t15_kl task_name=test ckpt_path=auto
    ;;
  3)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g002_t15_mse task_name=test ckpt_path=auto
    ;;
  4)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g005_t63_kl task_name=test ckpt_path=auto
    ;;
  5)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g005_t63_mse task_name=test ckpt_path=auto
    ;;
  6)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g005_t15_kl task_name=test ckpt_path=auto
    ;;
  7)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_g005_t15_mse task_name=test ckpt_path=auto
    ;;
  8)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u0005_t63_kl task_name=test ckpt_path=auto
    ;;
  9)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u0005_t63_mse task_name=test ckpt_path=auto
    ;;
  10)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u0005_t15_kl task_name=test ckpt_path=auto
    ;;
  11)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u0005_t15_mse task_name=test ckpt_path=auto
    ;;
  12)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u001_t63_kl task_name=test ckpt_path=auto
    ;;
  13)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u001_t63_mse task_name=test ckpt_path=auto
    ;;
  14)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u001_t15_kl task_name=test ckpt_path=auto
    ;;
  15)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d2_u001_t15_mse task_name=test ckpt_path=auto
    ;;
  16)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g002_t63_kl task_name=test ckpt_path=auto
    ;;
  17)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g002_t63_mse task_name=test ckpt_path=auto
    ;;
  18)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g002_t15_kl task_name=test ckpt_path=auto
    ;;
  19)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g002_t15_mse task_name=test ckpt_path=auto
    ;;
  20)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g005_t63_kl task_name=test ckpt_path=auto
    ;;
  21)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g005_t63_mse task_name=test ckpt_path=auto
    ;;
  22)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g005_t15_kl task_name=test ckpt_path=auto
    ;;
  23)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_g005_t15_mse task_name=test ckpt_path=auto
    ;;
  24)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u0005_t63_kl task_name=test ckpt_path=auto
    ;;
  25)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u0005_t63_mse task_name=test ckpt_path=auto
    ;;
  26)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u0005_t15_kl task_name=test ckpt_path=auto
    ;;
  27)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u0005_t15_mse task_name=test ckpt_path=auto
    ;;
  28)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u001_t63_kl task_name=test ckpt_path=auto
    ;;
  29)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u001_t63_mse task_name=test ckpt_path=auto
    ;;
  30)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u001_t15_kl task_name=test ckpt_path=auto
    ;;
  31)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d16_u001_t15_mse task_name=test ckpt_path=auto
    ;;
  32)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g002_t63_kl task_name=test ckpt_path=auto
    ;;
  33)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g002_t63_mse task_name=test ckpt_path=auto
    ;;
  34)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g002_t15_kl task_name=test ckpt_path=auto
    ;;
  35)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g002_t15_mse task_name=test ckpt_path=auto
    ;;
  36)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g005_t63_kl task_name=test ckpt_path=auto
    ;;
  37)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g005_t63_mse task_name=test ckpt_path=auto
    ;;
  38)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g005_t15_kl task_name=test ckpt_path=auto
    ;;
  39)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_g005_t15_mse task_name=test ckpt_path=auto
    ;;
  40)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u0005_t63_kl task_name=test ckpt_path=auto
    ;;
  41)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u0005_t63_mse task_name=test ckpt_path=auto
    ;;
  42)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u0005_t15_kl task_name=test ckpt_path=auto
    ;;
  43)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u0005_t15_mse task_name=test ckpt_path=auto
    ;;
  44)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u001_t63_kl task_name=test ckpt_path=auto
    ;;
  45)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u001_t63_mse task_name=test ckpt_path=auto
    ;;
  46)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u001_t15_kl task_name=test ckpt_path=auto
    ;;
  47)
    python -m src.run \
      seed=${SEED} data.num_workers=3 \
      experiment=${METHOD}/benchmark_hd/d64_u001_t15_mse task_name=test ckpt_path=auto
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac