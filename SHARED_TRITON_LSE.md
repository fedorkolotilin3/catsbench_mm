# Shared Triton LSE in this experiment

Baseline (`src/methods/dlight_sb.py`), MM (including exponential regularization),
and benchmark (`src/catsbench/benchmarks/base.py`) now call the same public
`catsbench.utils.lse_matmul` function for log-semiring matrix products.
MM's compatibility helper delegates to it under `torch.no_grad()`.
Baseline retains the custom Triton backward needed for training.

On CUDA the default backend is `triton`. The MM train/test shell scripts now
select that backend too. Do not set `LSE_BACKEND=cpu`, `any`, or `normalized`
when comparing this experiment's CUDA runs. CPU execution still uses the
shared PyTorch reference fallback because Triton requires CUDA.

Scalar/vector logsumexp reductions and the exponential regularization formula
are unchanged. The existing Triton kernel accumulates in float32 and converts
inputs to float32; this is not a float64 implementation or a fix for the
previous extreme-logit numerical issue. Existing result archives are unchanged.

# Validation

From this checkout, with the checkout root and `src` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src;$PWD\tests"
python -m unittest test_dlight_sb_mm test_mm_regularized test_shared_mm_lse -v
```

The CUDA tests compare forward/backward with a float64 PyTorch reference,
cover broadcasting, non-contiguous operands, exact zero support and MM's
no-grad helper, and check regularized MM loss descent and baseline backward.
They skip when CUDA is unavailable. No full experiment is launched.

# Transfer to the main repository

Port only the helper change in `src/methods/dlight_sb_mm.py`, the two
`scripts/*hdg_mm.sh` backend settings, and `tests/test_shared_mm_lse.py`.
Preserve any newer no-grad changes in the destination MM implementation.
No notebook, experiment configuration, model state or archive is required.
