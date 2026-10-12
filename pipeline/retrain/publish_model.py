"""Публикация модели: пара артефактов, указатель версии, сдвиг состояния окна."""

import json
import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.features import load_preproc, metrics_path
from pipeline.settings import (
    CURRENT_DATASET,
    KEEP_VERSIONS,
    LATEST_POINTER,
    MODELS_DIR,
)
from pipeline.train import cleanup_versions

logger = logging.getLogger(__name__)


def staged_model_path(version: str) -> Path:
    """Путь к черновику модели, который создал fit_catboost."""
    return MODELS_DIR / f".staging_model_{version}.cbm"


def final_model_path(version: str) -> Path:
    """Путь к опубликованному файлу модели."""
    return MODELS_DIR / f"model_{version}.cbm"


def publish_model(train_info: dict) -> dict:
    """Публикует обученную версию и только потом сдвигает состояние окна.

    Порядок шагов выбран так, чтобы повтор задачи и падение в середине не
    пропустили месяц: сначала модель и словарь встают на финальные имена,
    затем атомарно меняется latest.txt, затем новое окно заменяет current.parquet
    и только последним шагом сдвигается training_state. Если указатель уже
    указывает на эту версию, работа считается выполненной и не повторяется.
    Возвращает версию, признак публикации и список удалённых версий.
    """
    version = train_info["version"]
    if not train_info.get("promote"):
        raise RuntimeError(
            f"Версия {version} помечена как непубликуемая, публикация не выполняется"
        )

    pointer = MODELS_DIR / LATEST_POINTER
    if pointer.exists() and pointer.read_text(encoding="utf-8").strip() == version:
        logger.info("Версия %s уже активна, повторная публикация не нужна.", version)
        return {"version": version, "published": False, "removed_versions": []}

    staged = staged_model_path(version)
    final = final_model_path(version)
    if staged.exists():
        os.replace(staged, final)
    elif not final.exists():
        raise FileNotFoundError(f"Нет файла модели для публикации: {staged}")

    # Словарь должен стоять рядом с моделью до переключения указателя.
    dictionary = MODELS_DIR / f"preproc_{version}.json"
    if not dictionary.exists():
        raise FileNotFoundError(f"Нет словаря Ф2 для публикации: {dictionary}")
    if load_preproc(dictionary).get("version") != version:
        raise RuntimeError(f"Словарь {dictionary} относится к другой версии")

    metrics_file = metrics_path(version, MODELS_DIR)
    published_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    published_metrics["published_at"] = date.today().isoformat()
    metrics_file.write_text(
        json.dumps(published_metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    temporary_pointer = MODELS_DIR / f".{LATEST_POINTER}.{version}"
    temporary_pointer.write_text(version + "\n", encoding="utf-8")
    os.replace(temporary_pointer, pointer)
    logger.info("Опубликована версия %s (%s)", version, final)

    combined = {"version": version, "published": True, "removed_versions": []}
    if train_info.get("dropped_month") is not None:
        pending = Path(train_info.get("dataset_path") or "")
        if pending and pending.exists() and pending != CURRENT_DATASET:
            os.replace(pending, CURRENT_DATASET)
            logger.info("Окно обучения заменено на %s", CURRENT_DATASET)

    conn = get_conn()
    try:
        advance_training_state(conn, date.fromisoformat(train_info["batch_month"]))
    finally:
        conn.close()

    removed = cleanup_versions(
        MODELS_DIR,
        keep_count=KEEP_VERSIONS,
        keep=[version, train_info.get("current_version")],
    )
    combined["removed_versions"] = removed
    return combined
