#!/usr/bin/env bash
set -e

METHOD=${1:?Usage: $0 METHOD}
case "${METHOD}" in
  dlight_sb|dlight_sb_m|csbm|alpha_csbm|c2sbm) ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

PYTHON_BIN=${PYTHON_BIN:-/usr/local/bin/python3}

SEED=5
DIMS=(2 16 64)
BENCHMARKS=(g002 g005 u0005 u001)
VARIANTS=(t63_kl t63_mse t15_kl t15_mse)

for DIM in "${DIMS[@]}"; do
  if [[ "${METHOD}" == dlight_sb ]]; then
    EXPERIMENTS=
    for BENCHMARK in "${BENCHMARKS[@]}"; do
      EXPERIMENTS+=${EXPERIMENTS:+,}${METHOD}/benchmark_hd/d${DIM}_${BENCHMARK}
    done
  else
    for BENCHMARK in "${BENCHMARKS[@]}"; do
      EXPERIMENTS=
      for VARIANT in "${VARIANTS[@]}"; do
        EXPERIMENTS+=${EXPERIMENTS:+,}${METHOD}/benchmark_hd/d${DIM}_${BENCHMARK}_${VARIANT}
      done

      "${PYTHON_BIN}" -m src.run -m \
        hydra/launcher=submitit_local \
        hydra.launcher.timeout_min=525600 \
        hydra.launcher.gpus_per_node=1 \
        hydra.launcher.tasks_per_node=1 \
        hydra.launcher.cpus_per_task=2 \
        hydra.launcher.mem_gb=80 \
        seed=${SEED} data.num_workers=0 data.pin_memory=false \
        logger=csv \
        ckpt_path=null \
        experiment=${EXPERIMENTS}
    done
    continue
  fi

  "${PYTHON_BIN}" -m src.run -m \
      hydra/launcher=submitit_local \
      hydra.launcher.timeout_min=525600 \
      hydra.launcher.gpus_per_node=1 \
      hydra.launcher.tasks_per_node=1 \
      hydra.launcher.cpus_per_task=2 \
      hydra.launcher.mem_gb=80 \
      seed=${SEED} data.num_workers=0 data.pin_memory=false \
      logger=csv \
      ckpt_path=null \
      experiment=${EXPERIMENTS}
done
