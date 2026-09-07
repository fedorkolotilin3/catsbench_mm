"""
generate_dataset.py

Генерирует датасет для задачи EOT/SB на дискретных пространствах,
используя пакет `catsbench` (бенчмарк из статьи
"Entering the Era of Discrete Diffusion Models: A Benchmark for
Schrödinger Bridges and Entropic Optimal Transport").

Даёт:
    - две независимые выборки из маргиналов p_0(x_0) и p_1(x_1)
      (т.е. (x0, x1) ~ p_0(x0) p_1(x1), независимо друг от друга);
    - опционально — выборку из истинного EOT/SB-сопряжения
      (x0, x1) ~ p_0(x0) p*(x1 | x0), если нужен парный (coupled) датасет
      для валидации/оценки метода относительно ground truth.

Установка зависимости:
    pip install catsbench

Пример запуска:
    python generate_dataset.py \
        --benchmark-name hd_d2_s50_prior_gaussian_a0.02 \
        --n-samples 10000 \
        --seed 0 \
        --out-dir ./data
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        type=str,
        default="gregkseno/catsbench",
        help="HuggingFace репозиторий с предобученными бенчмарками.",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="hd_d2_s50_prior_gaussian_a0.02",
        help="Имя конкретного бенчмарка внутри репозитория "
        "(размерность D, число состояний S, референсный процесс и т.д.).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Число сэмплов в каждой из выборок p_0 и p_1.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed для воспроизводимости."
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Дополнительно сохранить парную выборку из истинного "
        "EOT/SB-сопряжения (x0, x1) ~ p_0(x0) p*(x1|x0), а не только "
        "независимые маргиналы.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data",
        help="Куда сохранить .npy файлы с выборками.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Импорт внутри main, чтобы скрипт понятно падал с подсказкой pip install,
    # если пакет не установлен.
    try:
        from catsbench import BenchmarkHD
    except ImportError as exc:
        raise ImportError(
            "Пакет `catsbench` не найден. Установите его командой:\n"
            "    pip install catsbench"
        ) from exc

    np.random.seed(args.seed)

    # init_benchmark=False пропускает тяжёлую инициализацию (например,
    # построение полного референсного процесса), если вам нужны только
    # сэмплы из маргиналов, а не точная плотность p*(x1|x0).
    # Если нужен --paired, инициализация обязательна.
    bench = BenchmarkHD.from_pretrained(
        args.repo_id,
        args.benchmark_name,
        init_benchmark=args.paired,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Независимые выборки из маргиналов: (x0, x1) ~ p_0(x0) p_1(x1) ---
    x0 = bench.sample_input(args.n_samples)   # [N, D]
    x1 = bench.sample_target(args.n_samples)  # [N, D]

    np.save(out_dir / "p0_samples.npy", np.asarray(x0))
    np.save(out_dir / "p1_samples.npy", np.asarray(x1))

    print(f"Сохранено {args.n_samples} сэмплов p_0 -> {out_dir / 'p0_samples.npy'}")
    print(f"Сохранено {args.n_samples} сэмплов p_1 -> {out_dir / 'p1_samples.npy'}")
    print(f"Форма x0: {np.asarray(x0).shape}, форма x1: {np.asarray(x1).shape}")

    # --- Опционально: парная выборка из истинного EOT/SB-сопряжения ---
    # (x0, x1) ~ p_0(x0) p*(x1 | x0) — полезно как ground truth для оценки
    # качества вашего метода (MM-алгоритма) относительно точного решения.
    if args.paired:
        x0_paired, x1_paired = bench.sample_input_target(args.n_samples)
        np.save(out_dir / "p0_paired.npy", np.asarray(x0_paired))
        np.save(out_dir / "p1_paired.npy", np.asarray(x1_paired))
        print(
            f"Сохранена парная выборка из ground-truth сопряжения "
            f"-> {out_dir / 'p0_paired.npy'}, {out_dir / 'p1_paired.npy'}"
        )


if __name__ == "__main__":
    main()