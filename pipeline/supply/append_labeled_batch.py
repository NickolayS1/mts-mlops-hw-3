"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from pathlib import Path

import pandas as pd
from airflow.exceptions import AirflowSkipException

from pipeline.db import copy_labeled_frame, get_conn, month_end_exclusive

logger = logging.getLogger(__name__)


def month_row_count(conn, batch_month) -> int:
    """Сколько наблюдений месяца уже лежит в витрине."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*)
            FROM labeled_transactions
            WHERE transaction_time >= %s AND transaction_time < %s
            """,
            (batch_month, month_end_exclusive(batch_month)),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def append_labeled_batch(batch_info: dict) -> dict:
    """Дописывает parquet батча в конец labeled_transactions.

    Читает путь из batch_info. Если наблюдения этого месяца в витрине уже есть,
    задача помечается skipped и ничего не пишет: COPY не проверяет дубли, а
    решение принимается по состоянию витрины, а не по файлу с хоста, поэтому
    потеря supply_batch.parquet между задачами не приводит к пропуску месяца.
    Возвращает месяц батча и число записанных строк.
    """
    batch_month = pd.Timestamp(batch_info["batch_month"])
    conn = get_conn()
    try:
        existing = month_row_count(conn, batch_month)
        if existing:
            raise AirflowSkipException(
                f"Месяц {batch_month.date()} уже в витрине ({existing} строк), дозапись не нужна."
            )

        batch_path = Path(batch_info["batch_path"])
        if not batch_path.exists():
            raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")
        frame = pd.read_parquet(batch_path)
        copy_labeled_frame(conn, frame)
    finally:
        conn.close()

    logger.info("Месяц %s добавлен в витрину, строк: %s", batch_month.date(), len(frame))
    return {"batch_month": batch_info["batch_month"], "rows": int(len(frame))}
