"""Create the fixed CPU comparison artifact once; reuse it on subsequent runs."""

import os
os.environ["LSE_BACKEND"] = "cpu"

from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from catsbench import BenchmarkHD, BenchmarkHDConfig
from src.methods.dlight_sb import DLightSB


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fixed_dlight_sb_mm.pt"


def prepare(path=DATA_PATH):
    path = Path(path)
    if path.exists():
        print(f"Reusing {path} (sha256={hashlib.sha256(path.read_bytes()).hexdigest()})")
        return path
    started = perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    config = BenchmarkHDConfig(
        dim=2, input_shape=(2,), num_categories=50, num_potentials=5,
        radius=5.0, alpha=0.02, num_timesteps=15, num_skip_steps=8,
        prior_type="gaussian", benchmark_type="gaussian", input_distribution="gaussian",
        num_val_samples=128, init_batch_size=128, reverse=False, tau=1.0,
        params_dtype="float64", eps=0.0,
    )
    bench = BenchmarkHD(config, device="cpu")

    def draw(fn, n):
        return torch.cat([fn(min(256, n - i)) for i in range(0, n, 256)])

    # Independent empirical marginals; no paired supervision is used in loss.
    payload = {name: draw(fn, 10_000) for name, fn in (
        ("train_x", bench.sample_input), ("train_y", bench.sample_target),
        ("test_x", bench.sample_input), ("test_y", bench.sample_target),
        ("oracle_y", bench.sample_target),
    )}
    # Initialization has its own seed and uses train only.
    torch.manual_seed(123)
    model = DLightSB(
        prior=bench.prior, dim=2, num_categories=50, num_potentials=32,
        num_timesteps=15, optimizer=partial(torch.optim.AdamW, lr=0.03, weight_decay=0.0),
        distr_init="samples", entropy_lambda=0.0,
    ).double()
    model.init_weights(payload["train_y"][torch.randperm(10_000)[:32]])
    payload["initial_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Priors use non-persistent buffers, so state_dict alone is insufficient.
    payload["prior_buffers"] = {k: v.detach().clone() for k, v in bench.prior.named_buffers()}
    torch.manual_seed(456)
    payload["conditions"] = payload["test_x"][torch.linspace(0, 9999, 16).long()]
    payload["conditional_truth"] = torch.stack([
        torch.cat([bench.sample(b) for b in x[None].repeat(512, 1).split(256)])
        for x in payload["conditions"]
    ])
    payload["metadata"] = {
        "benchmark": asdict(config), "data_seed": 42, "init_seed": 123,
        "conditional_seed": 456, "sampling_seed": 789, "num_potentials": 32,
        "dtype": "float64", "torch": str(torch.__version__),
        "numpy": str(np.__version__), "prepare_seconds": perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {**payload["metadata"], "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {path} (sha256={manifest['file_sha256']})")
    return path


def prepare_large(path=None, *, base_path=DATA_PATH, n_train=1_000_000):
    """Extend train only; retain the exact test data, prior and initialization."""
    path = Path(path) if path is not None else DATA_PATH.with_name("fixed_dlight_sb_mm_million.pt")
    base_path = Path(prepare(base_path))
    if path.resolve() == base_path.resolve():
        raise ValueError("The large dataset must use a separate file")
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    payload = torch.load(base_path, map_location="cpu", weights_only=True)
    if n_train <= max(len(payload["train_x"]), len(payload["train_y"])):
        raise ValueError("n_train must exceed the base train size")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["metadata"].get("base_sha256") != base_hash
                or len(saved["train_x"]) != n_train or len(saved["train_y"]) != n_train):
            raise ValueError("Existing large dataset has a different base or size")
        return path
    started = perf_counter()
    metadata = payload["metadata"]
    # The original benchmark parameters were generated before any data draws.
    if metadata["torch"] != str(torch.__version__) or metadata["numpy"] != str(np.__version__):
        raise ValueError("Reconstruct the benchmark with the original Torch/NumPy versions")
    torch.set_num_threads(4)
    torch.manual_seed(metadata["data_seed"])
    np.random.seed(metadata["data_seed"])
    bench = BenchmarkHD(BenchmarkHDConfig(**metadata["benchmark"]), device="cpu")
    for name, expected in payload["prior_buffers"].items():
        if not torch.equal(dict(bench.prior.named_buffers())[name], expected):
            raise ValueError("Reconstructed prior differs from the fixed benchmark")
    for name, fn in (("train_x", bench.sample_input), ("train_y", bench.sample_target)):
        size = len(payload[name])
        reconstructed = torch.cat([fn(min(256, size - i)) for i in range(0, size, 256)])
        if not torch.equal(reconstructed, payload[name]):
            raise ValueError("Reconstructed benchmark does not reproduce the base train")
    # An independent stream extends the original train, without regenerating test.
    torch.manual_seed(2026)
    for name, fn in (("train_x", bench.sample_input), ("train_y", bench.sample_target)):
        extra = n_train - len(payload[name])
        payload[name] = torch.cat([payload[name]] + [
            fn(min(256, extra - i)) for i in range(0, extra, 256)
        ])
    payload["metadata"] = dict(metadata, n_train=n_train, extension_seed=2026,
                               base_sha256=base_hash, prepare_seconds=perf_counter() - started)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {**payload["metadata"], "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    prepare()
