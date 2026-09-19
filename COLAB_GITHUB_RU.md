# catsbench_mm: GitHub, Google Colab и Google Drive

Проект содержит воспроизводимый baseline и единственный итоговый MM-эксперимент
для BenchmarkHD `d2_g002`.

## 1. Сохранение проекта в личный GitHub

Создайте на GitHub пустой репозиторий `catsbench_mm` без README, `.gitignore` и
лицензии. Из корня локального проекта запустите:

```bash
bash scripts/publish_github.sh git@github.com:YOUR_LOGIN/catsbench_mm.git
```

Для HTTPS вместо SSH:

```bash
bash scripts/publish_github.sh https://github.com/YOUR_LOGIN/catsbench_mm.git
```

Скрипт показывает все изменённые и новые файлы до коммита и ждёт подтверждения.
Исходный remote `origin` (`gregkseno/catsbench`) не меняется. Личный репозиторий
добавляется как `personal`, а состояние отправляется в ветку `catsbench-mm`.

Перед запуском при необходимости задайте Git-имя и email:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

Токены и пароли нельзя добавлять в репозиторий. Для HTTPS используйте GitHub
Personal Access Token через менеджер учётных данных; для SSH — SSH-ключ.

## 2. Развёртывание на новом Colab-сервере

В терминале Colab/VS Code:

```bash
cd /content
git clone --branch catsbench-mm https://github.com/fedorkolotilin3/catsbench_mm.git
cd catsbench_mm
bash scripts/setup_colab.sh
```

Скрипт через `uv` скачивает Python 3.12 и не зависит от версии системного Python
в образе Colab. По умолчанию устанавливаются версии PyTorch из исходного
окружения (`2.6.0`, CUDA 12.4). Чтобы вместо них поставить актуальную сборку с
PyPI:

```bash
PIN_TORCH=0 bash scripts/setup_colab.sh
```

`flash-attn` не нужен для `benchmark_hd` и по умолчанию не собирается. Для
экспериментов, где он действительно нужен:

```bash
INSTALL_FLASH_ATTN=1 bash scripts/setup_colab.sh
```

## 3. Полный baseline: обучение и тестирование на GPU

В настройках Colab сначала выберите GPU runtime, затем выполните:

```bash
bash scripts/run_baseline_train_test.sh
```

Сценарий сначала обучает обычный `DLightSB` градиентным спуском, затем загружает
`last.ckpt` и запускает полный test. `LSE_BACKEND=any` не требует сборки
локальных Triton-ядер.

## 4. Полный MM: обучение и тестирование на CPU

MM использует CPU и `float64`; наличие GPU на этом этапе не требуется:

```bash
bash scripts/run_mm_train_test.sh
```

По умолчанию используются все доступные CPU-потоки. Ограничить их число можно:

```bash
MM_NUM_THREADS=4 bash scripts/run_mm_train_test.sh
```

Оба сценария записывают обучение в `metrics/train/metrics.csv`, тестирование —
в `metrics/test/metrics.csv`, а модель — в `checkpoints/last.ckpt` внутри каталога
соответствующего Hydra-запуска.

## 5. Google Drive: сохранение результатов между сессиями

Сначала смонтируйте Drive кнопкой **Mount Drive** в расширении Colab для VS Code
или выполните в ячейке Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

После обучения сохраните каталог `logs` в
`MyDrive/catsbench_mm/logs`:

```bash
bash scripts/sync_google_drive.sh push
```

После подключения к новой машине восстановите результаты:

```bash
bash scripts/sync_google_drive.sh pull
```

Можно явно передать несколько относительных путей, например:

```bash
bash scripts/sync_google_drive.sh push logs data
bash scripts/sync_google_drive.sh pull logs data
```

Синхронизация добавочная: существующие файлы не удаляются. Другой каталог Drive
задаётся переменной окружения:

```bash
CATS_DRIVE_DIR=/content/drive/MyDrive/experiments/catsbench_mm \
  bash scripts/sync_google_drive.sh push logs
```

Метрики CSV находятся внутри соответствующего запуска по пути вида
`logs/runs/.../metrics/train/metrics.csv`, а checkpoints — в каталоге того же
запуска. Поэтому синхронизации `logs` достаточно для переноса результатов и
продолжения их анализа.
