#!/usr/bin/env python3
"""Run one HD benchmark method locally, four experiments at a time."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from omegaconf import OmegaConf


METHODS = ("dlight_sb", "dlight_sb_m", "csbm", "alpha_csbm", "c2sbm")
DIMS = (2, 16, 64)
BENCHMARKS = ("g002", "g005", "u0005", "u001")
VARIANTS = ("t63_kl", "t63_mse", "t15_kl", "t15_mse")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", PROJECT_ROOT / "logs" / "runs"))


def experiments(method: str) -> list[str]:
    if method == "dlight_sb":
        return [
            f"{method}/benchmark_hd/d{dim}_{benchmark}"
            for dim in DIMS
            for benchmark in BENCHMARKS
        ]
    return [
        f"{method}/benchmark_hd/d{dim}_{benchmark}_{variant}"
        for dim in DIMS
        for benchmark in BENCHMARKS
        for variant in VARIANTS
    ]


def latest_checkpoint(experiment: str, seed: int) -> Path | None:
    run_parent = RUNS_ROOT / experiment / str(seed)
    if not run_parent.is_dir():
        return None

    checkpoints = [
        run_dir / "checkpoints" / "last.ckpt"
        for run_dir in run_parent.iterdir()
        if run_dir.is_dir()
        and (run_dir / "checkpoints" / "last.ckpt").is_file()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda checkpoint: checkpoint.parent.parent.name)


def training_is_complete(checkpoint: Path) -> bool:
    run_dir = checkpoint.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return False

    try:
        config = OmegaConf.load(config_path)
        max_epochs = int(config.trainer.max_epochs)
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        completed_epochs = int(state["epoch"]) + 1
        del state
    except Exception as error:
        print(
            f"WARNING could not check completion of {checkpoint}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return False

    return max_epochs > 0 and completed_epochs >= max_epochs


def train(experiment: str, seed: int) -> None:
    checkpoint = latest_checkpoint(experiment, seed)
    if checkpoint is None:
        ckpt_override = "ckpt_path=null"
        print(f"START  {experiment} from scratch", flush=True)
    elif training_is_complete(checkpoint):
        print(f"SKIP   {experiment}: training already completed", flush=True)
        return
    else:
        checkpoint = checkpoint.resolve()
        ckpt_override = f"ckpt_path={checkpoint}"
        print(f"RESUME {experiment} from {checkpoint}", flush=True)

    command = [
        sys.executable,
        "-m",
        "src.run",
        f"experiment={experiment}",
        f"seed={seed}",
        "data.num_workers=0",
        "data.pin_memory=false",
        "logger=csv",
        ckpt_override,
    ]
    subprocess.run(command, check=True)
    print(f"DONE  {experiment}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        jobs = {
            executor.submit(train, experiment, args.seed): experiment
            for experiment in experiments(args.method)
        }
        for job in as_completed(jobs):
            experiment = jobs[job]
            try:
                job.result()
            except Exception as error:
                failures.append(experiment)
                print(f"FAILED {experiment}: {error}", file=sys.stderr, flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
