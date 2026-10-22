"""Run directly: short GPU verification without changing production kernel configs."""

# %% Load this checkout and measure each stage independently.
from pathlib import Path
import json
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / 'src')]
import torch
from catsbench import BenchmarkHD
from catsbench.lse_matmul.triton import triton_lse_matmul
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

report = {}
torch.manual_seed(64)


def measure(name, function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    report[name] = dict(seconds=time.perf_counter() - started,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    print(name, report[name], flush=True)
    (root / 'float64_launch_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result


# %% Compare real forward/backward launches with CPU double precision.
def check_accuracy():
    for dtype in (torch.float32, torch.float64):
        for exp2 in (False, True):
            a = torch.randn(2, 50, 50, device='cuda', dtype=dtype).requires_grad_()
            b = torch.randn(1, 50, 50, device='cuda', dtype=dtype).requires_grad_()
            ar = a.detach().cpu().double().requires_grad_()
            br = b.detach().cpu().double().requires_grad_()
            ref = (ar.unsqueeze(-1) + br.unsqueeze(-3)).logsumexp(-2)
            out = triton_lse_matmul(a, b, use_exp2=exp2)
            tol = 1e-11 if dtype == torch.float64 else 2e-5
            torch.testing.assert_close(out.cpu().double(), ref, atol=tol, rtol=tol)
            grad = torch.randn_like(out)
            out.backward(grad)
            ref.backward(grad.cpu().double())
            torch.testing.assert_close(a.grad.cpu().double(), ar.grad, atol=tol, rtol=tol)
            torch.testing.assert_close(b.grad.cpu().double(), br.grad, atol=tol, rtol=tol)
    return True


measure('forward_backward_accuracy', check_accuracy)

# %% Exercise GPU benchmark initialization and validation-time generation.
benchmark = measure('benchmark_gpu_initialization', lambda: BenchmarkHD.from_pretrained(
    'gregkseno/catsbench', subfolder='hd_d16_s50_prior_gaussian_a0.02',
    num_timesteps=15, init_benchmark=False, device='cuda', local_files_only=True))
assert benchmark.log_cp_cores.dtype == benchmark.prior.dtype == torch.float64
with torch.no_grad():
    measure('benchmark_sampling_128', lambda: benchmark.sample(benchmark.input_dataset[:128]))

# %% Run a single analytic MM update at the configured training batch size.
with initialize_config_dir(config_dir=str(root / 'configs'), version_base='1.1'):
    cfg = compose(config_name='config', overrides=['experiment=dlight_sb_mm/benchmark_hd/d16_g002'])
model = measure('model_initialization', lambda: instantiate(cfg.method).cuda())
model.init_weights(benchmark.input_dataset[:16])
batch = cfg.data.batch_size
x = benchmark.input_dataset[torch.randint(len(benchmark.input_dataset), (batch,), device='cuda')]
y = torch.randint(0, 50, (batch, 16), device='cuda')
with torch.no_grad():
    c = measure('model_log_c_full_batch', lambda: model.get_log_c(x))
    assert c.dtype == torch.float64 and torch.isfinite(c).all()
info = measure('mm_update_full_batch', lambda: model.mm_step(x, y))
assert all(torch.isfinite(p).all() for p in model.parameters())
report['batch_size'] = batch
report['mm_info'] = info
(root / 'float64_launch_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PASS', flush=True)
