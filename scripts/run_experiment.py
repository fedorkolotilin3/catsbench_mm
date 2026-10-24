"""Run one Hydra experiment and optionally export its complete run archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from hydra import compose, initialize_config_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], environment: dict[str, str], dry_run: bool) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )


def _archive_run(
    run_dir: Path,
    experiment: str,
    seed: int,
    precision: str,
    timestamp: str,
) -> Path:
    required = [
        run_dir / "checkpoints" / "last.ckpt",
        run_dir / ".hydra" / "config.yaml",
        run_dir / ".hydra" / "overrides.yaml",
        run_dir / "train.log",
        run_dir / "test.log",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot archive incomplete run: " + ", ".join(missing))

    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "seed": seed,
        "precision": precision,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "run_directory": str(run_dir.resolve()),
        "files": {
            path.relative_to(run_dir).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }

    exports_dir = PROJECT_ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    short_sha = manifest["git_commit"][:8]
    archive_name = (
        f"{_slug(experiment)}__seed-{seed}__{_slug(precision)}__"
        f"{timestamp}__{short_sha}.zip"
    )
    archive_path = exports_dir / archive_name
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for path in files:
            archive.write(path, Path("run") / path.relative_to(run_dir))
    return archive_path


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one configured catsbench experiment"
    )
    parser.add_argument(
        "experiment",
        help="Hydra experiment name, for example mm_simplex/benchmark_hd/d16_g002",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a verified ZIP after train and test",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Skip checkpoint testing; incompatible with --archive",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    args, overrides = parser.parse_known_args()
    if args.archive and args.train_only:
        parser.error("--archive requires the test stage")
    return args, overrides


def main() -> int:
    args, overrides = _parse_args()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(
            config_name="config",
            overrides=[f"experiment={args.experiment}", *overrides],
        )
    seed = int(config.seed)
    precision = str(config.trainer.get("precision", "32-true"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        PROJECT_ROOT
        / "logs"
        / "runs"
        / Path(args.experiment)
        / str(seed)
        / timestamp
    ).resolve()
    run_override = f"hydra.run.dir={run_dir.as_posix()}"

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment["PYTHONUNBUFFERED"] = "1"
    thread_count = str(os.cpu_count() or 1)
    environment.setdefault("OMP_NUM_THREADS", thread_count)
    environment.setdefault("MKL_NUM_THREADS", thread_count)
    environment.setdefault("OPENBLAS_NUM_THREADS", thread_count)

    common = [
        sys.executable,
        "-m",
        "src.run",
        f"experiment={args.experiment}",
        run_override,
        *overrides,
    ]
    _run([*common, "logger.csv.version=train"], environment, args.dry_run)

    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if not args.train_only:
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not create {checkpoint}")
        _run(
            [
                *common,
                "task_name=test",
                f"ckpt_path={checkpoint.as_posix()}",
                "logger.csv.version=test",
            ],
            environment,
            args.dry_run,
        )

    if args.archive and not args.dry_run:
        archive = _archive_run(
            run_dir,
            args.experiment,
            seed,
            precision,
            timestamp,
        )
        print(f"Archive: {archive}", flush=True)
    elif args.dry_run:
        print(f"Run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
