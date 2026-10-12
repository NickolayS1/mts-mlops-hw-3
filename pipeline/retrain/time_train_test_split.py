"""Временной train/test split скользящего окна."""

import logging

import pandas as pd

from pipeline.settings import SPLITS_DIR

logger = logging.getLogger(__name__)


def time_train_test_split(window_info: dict) -> dict:
    """Делит окно по времени: ранние месяцы в train, свежий месяц в test.

    Читает parquet окна из window_info и пишет сырые части в splits/. Случайный
    split не используется: месяцы идут по порядку, поэтому проверка честно
    оценивает модель на будущем относительно обучения месяце.
    Возвращает пути к частям и подписи месяцев.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(window_info["dataset_path"])
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    frame = frame.sort_values("transaction_time")
    periods = frame["transaction_time"].dt.to_period("M")
    latest = periods.max()
    train = frame.loc[periods < latest]
    test = frame.loc[periods == latest]
    if train.empty or test.empty:
        raise RuntimeError(
            f"Окно не делится по времени: train {len(train)}, test {len(test)}"
        )

    train_path = SPLITS_DIR / "train_raw.parquet"
    test_path = SPLITS_DIR / "test_raw.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    train_months = sorted(periods.loc[periods < latest].unique().astype(str))
    logger.info(
        "Train месяцы %s (%s строк), test %s (%s строк).",
        train_months,
        len(train),
        latest,
        len(test),
    )
    return {
        "train_raw_path": str(train_path),
        "test_raw_path": str(test_path),
        "dataset_path": window_info["dataset_path"],
        "batch_month": window_info["batch_month"],
        "dropped_month": window_info.get("dropped_month"),
        "train_months": ",".join(train_months),
        "test_month": str(latest),
    }
