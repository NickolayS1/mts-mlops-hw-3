"""Обучение CatBoost на текущем окне: кандидат в публикацию."""

import logging

import pandas as pd

from pipeline.settings import LATEST_POINTER, MODELS_DIR
from pipeline.train import new_version, read_metrics, train_model, write_pair

logger = logging.getLogger(__name__)


def current_version() -> str | None:
    """Версия из указателя модели. None — если указателя ещё нет."""
    pointer = MODELS_DIR / LATEST_POINTER
    if not pointer.exists():
        return None
    return pointer.read_text(encoding="utf-8").strip() or None


def fit_catboost(split_info: dict) -> dict:
    """Обучает модель-кандидата на train и считает метрики на свежем месяце.

    Читает splits/train_raw.parquet и splits/test_raw.parquet. Пишет пару
    модель+словарь и файл метрик в каталог моделей под новой версией, но
    latest.txt не трогает: активной версией кандидат станет только после
    проверки улучшения. Повторный запуск создаёт новую версию и публикацию
    не сдвигает, пока предыдущая не опубликована.
    Возвращает версию кандидата и метаданные окна для следующих задач.
    """
    train_frame = pd.read_parquet(split_info["train_raw_path"])
    test_frame = pd.read_parquet(split_info["test_raw_path"])
    version = new_version()
    model, payload = train_model(train_frame, test_frame, MODELS_DIR, version=version)
    write_pair(model, payload["preproc"], payload["metrics"], MODELS_DIR, staging=True)

    active = current_version()
    previous = read_metrics(active, MODELS_DIR) if active else None
    if active and previous is None:
        logger.warning("У активной версии %s нет файла метрик, сравнение невозможно.", active)
    return {
        "version": version,
        "current_version": active,
        "metrics": payload["metrics"],
        "previous_metrics": previous,
        "batch_month": split_info["batch_month"],
        "dropped_month": split_info.get("dropped_month"),
        "train_months": split_info["train_months"],
        "test_month": split_info["test_month"],
        # Публикация переносит это окно в current.parquet, поэтому путь
        # обязателен: без него сдвиг окна молча не произошёл бы.
        "dataset_path": split_info["dataset_path"],
    }
