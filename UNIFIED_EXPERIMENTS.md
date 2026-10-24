# Unified experiment workflow

The repository exposes three algorithm families through Hydra configs

- `dlight_sb` is the gradient baseline and keeps optimizer selection in config
- `dlight_sb_sgd` is the plain SGD baseline with a large configurable learning rate
- `dlight_sb_mm` is ordinary MM with optional regularization clipping and post-step projection
- `dlight_sb_mm_simplex` is constrained MM with compiled log-domain bisection

The gradient baseline and ordinary MM also expose `freeze_beta` for the historical fixed-mixture experiment

Run an experiment with one short command

```bash
python scripts/run_experiment.py dlight_sb_mm/benchmark_hd/d16_g002
```

Run the long unregularized SGD baseline

```bash
python scripts/run_experiment.py dlight_sb_sgd/benchmark_hd/d16_g002 --archive
```

Run train then test and create a reproducible archive

```bash
python scripts/run_experiment.py mm_regularized/benchmark_hd/d16_g002 --archive
```

Run simplex MM

```bash
python scripts/run_experiment.py mm_simplex/benchmark_hd/d16_g002 --archive
```

Hydra overrides may be appended directly

```bash
python scripts/run_experiment.py dlight_sb_mm/benchmark_hd/d16_g002 trainer.max_steps=10
```

Every archive name includes the experiment name seed precision timestamp and Git commit

The archive contains the resolved Hydra config overrides logs metrics checkpoints and a SHA-256 manifest

## Ordinary MM parameters

- `regularization_lambda_alpha`
- `regularization_lambda_cores`
- `clip_updates`
- `min_component_value`
- `project_to_simplex_after_step`
- `compile_update_kernels`
- `freeze_beta`

Setting both regularization coefficients to zero gives unregularized MM

Post-step simplex projection is allowed only when both coefficients are zero because the regularized objective is not scale invariant

## Simplex MM parameters

- `bisection_steps`
- `compile_simplex_solver`
- `compile_update_kernels`

CUDA uses fixed-shape `torch.compile` kernels with eager CPU fallbacks
