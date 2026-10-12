"""Поставка следующего месяца в витрину.

Месяц берётся из training_state.next_batch_month, то есть ровно тот, который
DAG переобучения возьмёт следующим. Витрина поэтому опережает окно обучения
на один месяц, и prune_old_months начинает удалять старые наблюдения.
"""

import calendar
import logging

import pandas as pd
from airflow.exceptions import AirflowSkipException

from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive, read_training_state
from pipeline.settings import SAMPLE_SIZE, SUPPLY_BATCH, TRAINING_DIR

logger = logging.getLogger(__name__)


def shift_into_month(frame: pd.DataFrame, month_start) -> pd.DataFrame:
    """Переносит transaction_time в заданный месяц, сохраняя день, час и минуту.

    Если в целевом месяце нет такого дня, день обрезается до последнего числа.
    Секунды обнуляются: исходные отметки времени идут с точностью до минуты.
    Исходный frame не меняется.
    """
    shifted = frame.copy()
    moments = pd.to_datetime(shifted["transaction_time"])
    year, month = int(month_start.year), int(month_start.month)
    last_day = calendar.monthrange(year, month)[1]
    shifted["transaction_time"] = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": moments.dt.day.clip(upper=last_day),
            "hour": moments.dt.hour,
            "minute": moments.dt.minute,
        }
    )
    return shifted


def newest_month_key(conn) -> str | None:
    """Возвращает «YYYY-MM» самого нового наблюдения витрины.

    Формат даты ровно 19 символов, поэтому лексикографический максимум строки
    совпадает с хронологическим и отдельный индекс для этого не нужен.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT max(transaction_time)::text FROM labeled_transactions")
        row = cur.fetchone()
    newest = row[0] if row else None
    return newest[:7] if newest else None


def sample_next_month() -> dict:
    """Готовит батч целевого месяца из уже лежащих в витрине строк.

    Читает указатель месяца из training_state и случайные SAMPLE_SIZE строк
    витрины, переносит их время в целевой месяц и пишет supply_batch.parquet.
    Витрину не меняет, поэтому повтор задачи безопасен: батч в витрину
    добавляет append_labeled_batch, а он проверяет месяц.
    Если целевой месяц ушёл за последний месяц датасета, задача помечается
    skipped: переносить время из будущего не из чего, и дальше симуляция
    поставки не идёт.
    Возвращает путь к батчу, целевой месяц и число строк.
    """
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        state = read_training_state(conn)
        target_month = state["next_batch_month"]
        target_end = month_end_exclusive(target_month)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM labeled_transactions WHERE transaction_time >= %s AND transaction_time < %s",
                (target_month, target_end),
            )
            already_present = cur.fetchone()[0]

        newest_month = newest_month_key(conn)
        target_key = target_month.strftime("%Y-%m")
        if newest_month is None:
            raise RuntimeError("Витрина labeled_transactions пуста. Сначала должен отработать data_loader.")
        if target_key > newest_month:
            raise AirflowSkipException(
                f"Месяц {target_key} позже последнего месяца датасета ({newest_month}): "
                "размеченных данных для поставки больше нет."
            )

        columns = ", ".join(LABELED_COLUMNS)
        frame = pd.read_sql(
            f"""
            SELECT {columns}
            FROM labeled_transactions
            ORDER BY random()
            LIMIT %s
            """,
            conn,
            params=(SAMPLE_SIZE,),
        )
    finally:
        conn.close()

    if frame.empty:
        raise RuntimeError("Витрина labeled_transactions пуста. Сначала должен отработать data_loader.")

    batch = shift_into_month(frame, target_month)
    batch.to_parquet(SUPPLY_BATCH, index=False)
    logger.info(
        "Батч %s: %s строк из %s, время %s — %s, строк этого месяца в витрине уже %s",
        target_month,
        len(batch),
        len(frame),
        batch["transaction_time"].min(),
        batch["transaction_time"].max(),
        already_present,
    )
    return {
        "batch_path": str(SUPPLY_BATCH),
        "batch_month": target_month.isoformat(),
        "rows": int(len(batch)),
        "already_present": int(already_present),
    }
