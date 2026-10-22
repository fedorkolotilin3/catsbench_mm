# CatSBench DLightSB

Исследовательский репозиторий для DLightSB на высокоразмерном дискретном бенчмарке CatSBench

В репозитории оставлены четыре реализации

- `DLightSB` с AdamW
- `DLightSB-M`
- `DLightSBMM` с блочными MM-обновлениями
- `DLightSBMMRegularized` с экспоненциальной регуляризацией

Все реализации используют общий `catsbench.lse_matmul` и один Triton backend на CUDA. CPU backend остаётся эталоном для проверки точности

## Структура

```text
configs/             Hydra конфиги DLightSB и HD benchmark
notebooks/           запуск и сравнение архивов экспериментов
scripts/             локальные Colab и SLURM запуски
src/catsbench/       HD benchmark метрики prior и LSE matmul
src/data/            BenchmarkDataModule и batch
src/methods/         DLightSB и MM
src/metrics/         метрики benchmark_hd
src/plotter/         графики benchmark_hd
tests/               проверки MM Triton checkpoint и resume
```

## Установка

```bash
conda env update -f environment.yml
conda activate catsbench
```

Для Colab или обычного Linux-окружения

```bash
bash scripts/setup_colab.sh
```

## Запуск

```bash
python -m src.run experiment=dlight_sb/benchmark_hd/d16_g002
python -m src.run experiment=dlight_sb_m/benchmark_hd/d16_g002_t15_kl
python -m src.run experiment=dlight_sb_mm/benchmark_hd/d16_g002
python -m src.run experiment=mm_regularized/benchmark_hd/d16_g002
```

Тест последнего checkpoint для того же пресета

```bash
python -m src.run task_name=test ckpt_path=auto experiment=dlight_sb_mm/benchmark_hd/d16_g002
```

Готовые train и test сценарии

```bash
bash scripts/run_baseline_train_test.sh --experiment dlight_sb/benchmark_hd/d2_g002
bash scripts/run_mm_train_test.sh --device gpu --experiment dlight_sb_mm/benchmark_hd/d2_g002
```

## Ветки экспериментов

- `master` основная реализация
- `experiment/mm-simplex` симплексная параметризация
- `experiment/mm-simplex-newton` симплексная параметризация с шагом Ньютона
- `experiment/mm-unregularized` обычный MM без регуляризации
- `experiment/mm-exp-regularized` MM с экспоненциальной регуляризацией
- `experiment/mm-exp-regularized-split-weights` регуляризация с разными весами параметров

Переключение версии

```bash
git switch experiment/mm-exp-regularized
```

## Ноутбуки

- `run_dlight_sb_pairs_local.ipynb` локальный запуск baseline и MM
- `run_dlight_sb_pairs_colab.ipynb` тот же сценарий для Colab
- `analyze_dlight_sb_pair_local.ipynb` локальное сравнение двух или более архивов
- `analyze_dlight_sb_pair.ipynb` версия анализа для Colab
- `benchmark_usage.ipynb` интерфейс HD benchmark
- `benchmark_baselines.ipynb` эталонные baseline расчёты
- `adamw_from_mm.ipynb` диагностический запуск AdamW из параметров MM

Подробности находятся в `DOCUMENTATION_RU.md` и `COLAB_GITHUB_RU.md`

## Проверка

```bash
python -m unittest discover -s tests -v
```

Лицензия проекта находится в `LICENSE.md`
