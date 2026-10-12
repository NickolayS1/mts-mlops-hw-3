"""Обучение стартовой модели на первом окне датасета.

Стартовая пара нужна, чтобы сервис скоринга работал до первого переобучения.
Схема признаков и параметры CatBoost берутся из pipeline.train, поэтому
стартовая модель не отличается от опубликованных DAG.
"""

import logging

import pandas as pd

from pipeline.settings import CURRENT_DATASET, MODELS_DIR
from pipeline.train import train_model, write_pair

logger = logging.getLogger(__name__)

STARTER_VERSION = "initial"
LATEST_POINTER = "latest.txt"


def latest_month_split(window: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит окно по времени: ранние месяцы в train, самый свежий месяц в test."""
    moments = pd.to_datetime(window["transaction_time"])
    periods = moments.dt.to_period("M")
    latest = periods.max()
    train_frame = window.loc[periods < latest]
    test_frame = window.loc[periods == latest]
    if train_frame.empty or test_frame.empty:
        raise RuntimeError(
            f"Окно не делится по времени: train {len(train_frame)}, test {len(test_frame)}"
        )
    logger.info("Train месяцы %s, test месяц %s", sorted(periods.loc[periods < latest].unique().astype(str)), latest)
    return train_frame, test_frame


def starter_published() -> bool:
    """Проверяет, что стартовая пара и указатель уже на месте."""
    pointer = MODELS_DIR / LATEST_POINTER
    return (
        (MODELS_DIR / f"model_{STARTER_VERSION}.cbm").exists()
        and (MODELS_DIR / f"preproc_{STARTER_VERSION}.json").exists()
        and pointer.exists()
    )


def train_starter() -> str:
    """Обучает и публикует стартовую модель, если её ещё нет.

    Читает current.parquet из окна обучения. Повторный запуск при готовой паре
    ничего не делает, поэтому указатель модели не откатывается на initial.
    Возвращает версию, на которую указывает latest.txt.
    """
    pointer = MODELS_DIR / LATEST_POINTER
    if starter_published():
        logger.info("Стартовая пара уже опубликована, обучение пропущено.")
        return pointer.read_text(encoding="utf-8").strip()

    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет окна обучения: {CURRENT_DATASET}")
    window = pd.read_parquet(CURRENT_DATASET)
    train_frame, test_frame = latest_month_split(window)

    model, payload = train_model(train_frame, test_frame, MODELS_DIR, version=STARTER_VERSION)
    write_pair(model, payload["preproc"], payload["metrics"], MODELS_DIR)
    # Указатель последним: скоринг не должен увидеть модель без словаря.
    (MODELS_DIR / LATEST_POINTER).write_text(STARTER_VERSION + "\n", encoding="utf-8")
    logger.info("Стартовая пара опубликована в %s", MODELS_DIR)
    return STARTER_VERSION
