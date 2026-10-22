# DLightSB и MM в CatSBench

## Назначение репозитория

Репозиторий содержит только DLightSB-семейство и необходимые ему части CatSBench HD benchmark. Image и toy задачи а также CSBM Alpha-CSBM C2SBM VQGAN и IMF удалены

Рабочий контур включает загрузку benchmark обучение тестирование метрики визуализацию checkpoint архивы и анализ нескольких запусков

## Методы

| Конфиг | Класс | Назначение |
|---|---|---|
| `dlight_sb` | `DLightSB` | градиентное обучение AdamW |
| `dlight_sb_m` | `DLightSB_M` | DLightSB с KL и MSE коэффициентами |
| `dlight_sb_mm` | `DLightSBMM` | блочные MM-обновления |
| `mm_regularized` | `DLightSBMMRegularized` | MM с экспоненциальной регуляризацией |

Код методов находится в `src/methods`. Общий log-sum-exp матричных произведений находится в `src/catsbench/lse_matmul`

## Данные и benchmark

`BenchmarkDataModule` загружает `BenchmarkHD` из `gregkseno/catsbench`. Основные параметры задаются через `configs/data/benchmark_hd.yaml`

- `dim` размерность пространства
- `num_categories` число категорий координаты
- `num_timesteps` число шагов reference process
- `batch_size` размер train batch
- `val_batch_size` размер validation и test batch
- `num_train_batches` виртуальная длина train epoch

Готовые пресеты используют размерности 2 16 и 64. Baseline DLightSB и DLightSB-M содержат gaussian и uniform варианты. MM-пресеты оставлены для gaussian alpha 0.02 в размерностях 2 16 и 64

## Hydra

Главная точка входа

```bash
python -m src.run experiment=<имя пресета>
```

Примеры

```bash
python -m src.run experiment=dlight_sb/benchmark_hd/d2_g002
python -m src.run experiment=dlight_sb_m/benchmark_hd/d16_u001_t63_mse
python -m src.run experiment=dlight_sb_mm/benchmark_hd/d16_g002
python -m src.run experiment=mm_regularized/benchmark_hd/d16_g002
```

Любое поле можно переопределить из командной строки

```bash
python -m src.run \
  experiment=dlight_sb_mm/benchmark_hd/d16_g002 \
  trainer.max_steps=100 \
  data.batch_size=8192 \
  seed=7
```

По умолчанию используются CSVLogger и TensorBoardLogger. Результаты пишутся в `logs/runs/<experiment>/<seed>/<timestamp>`

## Checkpoint и resume

Периодические checkpoints создаёт `PeriodicCheckpoint`. Последний checkpoint сохраняется в каталоге конкретного запуска

Продолжение обучения

```bash
python -m src.run \
  experiment=dlight_sb_mm/benchmark_hd/d16_g002 \
  ckpt_path=/path/to/run/checkpoints/last.ckpt
```

Тестирование

```bash
python -m src.run \
  task_name=test \
  experiment=dlight_sb_mm/benchmark_hd/d16_g002 \
  ckpt_path=/path/to/run/checkpoints/last.ckpt
```

`ckpt_path=auto` выбирает последний сохранённый checkpoint соответствующего пресета и seed

## Метрики и графики

`BenchmarkHDMetricsCallback` считает unconditional и conditional shape score trend score classifier two sample test и trajectory KL когда метод предоставляет transition logits

`BenchmarkHDPlotterCallback` строит выборки и траектории. При CSV logger изображения сохраняются внутри каталога запуска

Ноутбуки анализа принимают два или более zip-архива. Они показывают train и validation кривые checkpoint динамику сырые параметры канонически нормированные веса компонент и нормированные вероятности внутри ядер. Ядра сортируются по весу компоненты

## LSE backend

Все DLightSB реализации вызывают общий `catsbench.utils.lse_matmul`

- `triton` используется на CUDA
- `cpu` служит точным эталоном
- `normalized` и `any` доступны для диагностических сравнений

Backend выбирается полем `prior.implementation` или переменной окружения `LSE_BACKEND`

## Скрипты

| Скрипт | Назначение |
|---|---|
| `setup_colab.sh` | создать окружение и установить зависимости |
| `run_colab.sh` | запустить Hydra команду в созданном окружении |
| `run_baseline_train_test.sh` | обучить и проверить DLightSB |
| `run_mm_train_test.sh` | обучить и проверить MM |
| `train_hdg.sh` | SLURM серия baseline HD |
| `test_hdg.sh` | SLURM тест baseline HD |
| `train_hdg_mm.sh` | SLURM обучение MM |
| `test_hdg_mm.sh` | SLURM тест MM |
| `train_hdg_local.py` | локальный параллельный запуск DLightSB или DLightSB-M |
| `publish_github.sh` | публикация подготовленной ветки |
| `sync_google_drive.sh` | синхронизация архивов экспериментов |

## Ноутбуки

Локальный рабочий сценарий находится в `notebooks/run_dlight_sb_pairs_local.ipynb`. Colab сценарий находится в `notebooks/run_dlight_sb_pairs_colab.ipynb`

Локальный анализ находится в `notebooks/analyze_dlight_sb_pair_local.ipynb`. Colab анализ находится в `notebooks/analyze_dlight_sb_pair.ipynb`

Ячейки загрузки обучения обработки и визуализации разделены чтобы отдельные этапы можно было перезапускать независимо

## Проверка изменений

Полный CPU набор тестов

```bash
python -m unittest discover -s tests -v
```

Проверка композиции Hydra без запуска обучения

```bash
python -m src.run --cfg job experiment=dlight_sb/benchmark_hd/d2_g002
python -m src.run --cfg job experiment=dlight_sb_mm/benchmark_hd/d2_g002
python -m src.run --cfg job experiment=mm_regularized/benchmark_hd/d2_g002
```

Triton тесты требуют CUDA. Если CUDA отсутствует они пропускаются штатно
