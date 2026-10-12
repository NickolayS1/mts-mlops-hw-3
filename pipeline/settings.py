"""Пути общих каталогов и параметры окна обучения.

Значения по умолчанию рассчитаны на запуск из Airflow, где рабочий каталог —
/opt/airflow, а каталоги моделей и данных смонтированы снаружи. Для локальной
проверки пути переопределяются переменными окружения.
"""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
DQ_REPORTS_DIR = Path(os.getenv("DQ_REPORTS_DIR", "/opt/airflow/dq_reports"))

# Окно обучения: pending_window.parquet копится до публикации модели,
# current.parquet читает обучение. Разделение не даёт упавшему прогону
# сдвинуть окно дважды.
CURRENT_DATASET = TRAINING_DIR / "current.parquet"
PENDING_WINDOW = TRAINING_DIR / "pending_window.parquet"
BATCHES_DIR = TRAINING_DIR / "batches"
SPLITS_DIR = TRAINING_DIR / "splits"
SUPPLY_BATCH = TRAINING_DIR / "supply_batch.parquet"

WINDOW_MONTHS = 3
HISTORY_YEARS = 1
SAMPLE_SIZE = 100_000

# Публикация версии: указатель активной модели, допуск сравнения метрик и
# глубина хранения пар артефактов. Используется задачами переобучения.
LATEST_POINTER = "latest.txt"
# Запас на шум: окна соседних запусков пересекаются, поэтому ROC-AUC меняется
# в третьем знаке. Версия хуже предыдущей на большую величину не публикуется.
METRIC_TOLERANCE = 0.005
# Сколько пар артефактов держать для отката: текущая и предыдущая.
KEEP_VERSIONS = 2
