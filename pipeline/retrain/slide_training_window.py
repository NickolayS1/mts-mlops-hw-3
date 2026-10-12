"""Сдвиг скользящего окна обучения на один месяц вперёд."""

import logging

import pandas as pd

from pipeline.settings import CURRENT_DATASET, PENDING_WINDOW, WINDOW_MONTHS

logger = logging.getLogger(__name__)


def drop_oldest_month(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Period]:
    """Убирает из окна ровно один самый старый месяц.

    Возвращает срез и удалённый месяц. Отбрасывается один период, а не все
    строки с минимальным месяцем: иначе при повторе задачи окно потеряло бы
    лишний месяц и стало бы короче WINDOW_MONTHS.
    """
    periods = frame["transaction_time"].dt.to_period("M")
    oldest = periods.min()
    return frame.loc[periods != oldest].sort_values("transaction_time"), oldest


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к окну и выбрасывает самый старый.

    Читает current.parquet и parquet батча. Новое окно пишется в
    pending_window.parquet и current.parquet не заменяет: подмена происходит
    только после публикации модели, поэтому падение до публикации не сдвигает
    окно. Если месяц батча уже в окне, окно возвращается без изменений —
    повтор задачи не съедает ещё один месяц.
    Returns: путь к новому окну, месяц батча и удалённый месяц.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет текущего окна обучения: {CURRENT_DATASET}")

    batch_month = pd.Period(batch_info["batch_month"], freq="M")
    current = pd.read_parquet(CURRENT_DATASET)
    batch = pd.read_parquet(batch_info["batch_path"])
    current["transaction_time"] = pd.to_datetime(current["transaction_time"])
    batch["transaction_time"] = pd.to_datetime(batch["transaction_time"])

    present = set(current["transaction_time"].dt.to_period("M").unique())
    if batch_month in present:
        logger.warning(
            "Месяц %s уже в окне: окно не сдвигается, повтор задачи безопасен.",
            batch_month,
        )
        return {
            "dataset_path": str(CURRENT_DATASET),
            "batch_month": batch_info["batch_month"],
            "dropped_month": None,
        }

    combined = pd.concat([current, batch], ignore_index=True)
    updated, dropped = drop_oldest_month(combined)
    if updated.empty:
        raise RuntimeError("После сдвига окно пустое, обучение невозможно")

    updated.to_parquet(PENDING_WINDOW, index=False)
    months = sorted(updated["transaction_time"].dt.to_period("M").unique().astype(str))
    logger.info(
        "Окно сдвинуто: %s строк, месяцы %s, удалён %s",
        len(updated),
        months,
        dropped,
    )
    if len(months) > WINDOW_MONTHS:
        logger.warning(
            "В окне %s месяцев вместо %s: проверьте пропущенные месяцы витрины.",
            len(months),
            WINDOW_MONTHS,
        )
    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_info["batch_month"],
        "dropped_month": str(dropped),
    }
