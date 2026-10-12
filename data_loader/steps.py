"""Пошаговая инициализация витрины: схема, CSV, состояние окна, окно обучения.

Все шаги идемпотентны и проверяют свои условия сами, поэтому повторный запуск
контейнера ничего не перезаписывает и не сбрасывает указатель месяца.
"""

import logging
import os
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta

from pipeline.db import LABELED_COLUMNS, copy_labeled_frame, get_conn, month_end_exclusive
from pipeline.settings import (
    CURRENT_DATASET,
    LATEST_POINTER,
    MODELS_DIR,
    TRAINING_DIR,
    WINDOW_MONTHS,
)

logger = logging.getLogger(__name__)

CSV_PATH = Path(os.getenv("TRAIN_DATA_CSV", "/data/train.csv"))

# Каталог сервиса скоринга, куда кладётся начальная версия модели.
# Пустое значение выключает зеркалирование.
BOOTSTRAP_DIR = Path(value) if (value := os.getenv("BOOTSTRAP_DIR")) else None

# Первое окно обучения — три месяца от начала датасета. Дальше окно двигает DAG.
INITIAL_WINDOW_START = date(2019, 1, 1)

CONNECT_ATTEMPTS = 30
CONNECT_PAUSE_SECONDS = 2

LABELED_DDL = """
CREATE TABLE IF NOT EXISTS labeled_transactions (
    transaction_time TIMESTAMP NOT NULL,
    merch TEXT,
    cat_id TEXT,
    amount DOUBLE PRECISION,
    name_1 TEXT,
    name_2 TEXT,
    gender TEXT,
    street TEXT,
    one_city TEXT,
    us_state TEXT,
    post_code INTEGER,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    population_city INTEGER,
    jobs TEXT,
    merchant_lat DOUBLE PRECISION,
    merchant_lon DOUBLE PRECISION,
    target SMALLINT
)
"""

TRAINING_STATE_DDL = """
CREATE TABLE IF NOT EXISTS training_state (
    id INT PRIMARY KEY,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    next_batch_month DATE NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

TIME_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS labeled_transactions_time_idx
ON labeled_transactions (transaction_time)
"""


def connect() -> psycopg2.extensions.connection:
    """Ждёт Postgres: контейнер может подниматься дольше сервиса."""
    last_error = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return get_conn()
        except psycopg2.OperationalError as exc:
            last_error = exc
            logger.info("Postgres ещё не готов (%s/%s): %s", attempt, CONNECT_ATTEMPTS, exc)
            time.sleep(CONNECT_PAUSE_SECONDS)
    raise last_error


def ensure_schema(conn) -> None:
    """Создаёт витрину и состояние окна, если postgres/init их не создал.

    Повторный вызов безопасен: операторы с IF NOT EXISTS, существующие данные
    и указатель месяца не меняются.
    """
    with conn.cursor() as cur:
        cur.execute(LABELED_DDL)
        cur.execute(TRAINING_STATE_DDL)
        cur.execute(TIME_INDEX_DDL)
    conn.commit()
    logger.info("Схема витрины проверена.")


def row_count(conn) -> int:
    """Количество строк в витрине: по нему решается, нужна ли загрузка CSV."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM labeled_transactions")
        return cur.fetchone()[0]


def load_csv(conn) -> int:
    """Копирует размеченный CSV в витрину, если она пуста.

    Витрина заполняется один раз: при непустой таблице CSV не читается, поэтому
    перезапуск контейнера не удваивает наблюдения.
    """
    if row_count(conn) > 0:
        logger.info("Витрина уже заполнена, загрузка CSV пропущена.")
        return 0
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Не найден размеченный CSV: {CSV_PATH}")

    started = time.time()
    frame = pd.read_csv(CSV_PATH, usecols=LABELED_COLUMNS)
    copy_labeled_frame(conn, frame)
    logger.info(
        "CSV загружен: %s строк за %.1f с, время %s — %s",
        len(frame),
        time.time() - started,
        frame["transaction_time"].min(),
        frame["transaction_time"].max(),
    )
    return len(frame)


def ensure_training_state(conn) -> bool:
    """Ставит начальное окно обучения, если указателя месяца ещё нет.

    Существующую строку не трогает: DAG мог уже сдвинуть окно, и сброс
    указателя заставил бы переобучаться на старых месяцах.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM training_state WHERE id = 1")
        if cur.fetchone()[0]:
            logger.info("training_state уже заполнен, указатель месяца сохраняется.")
            return False
        window_end = INITIAL_WINDOW_START + relativedelta(months=WINDOW_MONTHS - 1)
        cur.execute(
            """
            INSERT INTO training_state (id, window_start, window_end, next_batch_month)
            VALUES (1, %s, %s, %s)
            """,
            (INITIAL_WINDOW_START, window_end, month_end_exclusive(window_end)),
        )
    conn.commit()
    logger.info(
        "Начальное окно %s — %s, первый батч DAG %s",
        INITIAL_WINDOW_START,
        window_end,
        month_end_exclusive(window_end),
    )
    return True


def materialize_window(conn) -> Path:
    """Сохраняет первое окно обучения в current.parquet для DAG.

    Файл не перезаписывается: окно дальше двигает публикация модели, и повторная
    материализация откатила бы уже обученное окно назад.
    """
    if CURRENT_DATASET.exists():
        logger.info("Окно обучения уже есть: %s", CURRENT_DATASET)
        return CURRENT_DATASET

    with conn.cursor() as cur:
        cur.execute("SELECT window_start, window_end FROM training_state WHERE id = 1")
        window_start, window_end = cur.fetchone()
    columns = ", ".join(LABELED_COLUMNS)
    frame = pd.read_sql(
        f"""
        SELECT {columns}
        FROM labeled_transactions
        WHERE transaction_time >= %s AND transaction_time < %s
        ORDER BY transaction_time
        """,
        conn,
        params=(window_start, month_end_exclusive(window_end)),
    )
    if frame.empty:
        raise RuntimeError(f"В витрине нет наблюдений за окно {window_start} — {window_end}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CURRENT_DATASET, index=False)
    logger.info(
        "Окно обучения сохранено: %s строк, %s — %s",
        len(frame),
        frame["transaction_time"].min(),
        frame["transaction_time"].max(),
    )
    return CURRENT_DATASET


def mirror_starter_to_bootstrap() -> list[str]:
    """Кладёт стартовую версию в каталог, откуда её читает сервис скоринга.

    Общий каталог моделей и каталог сервиса живут в разных томах, поэтому
    сервису нужна своя копия: его том может остаться пустым после пересоздания,
    а обучение стартовой модели идёт в общем каталоге. Копируется именно версия
    initial, а не текущая опубликованная: сервис подхватит её только если
    указателя в своём томе нет, то есть когда публикаций ещё не было.
    Указатель пишется последним. Повторный вызов перезаписывает те же файлы.
    """
    if BOOTSTRAP_DIR is None:
        logger.info("BOOTSTRAP_DIR не задан, зеркалирование начальной версии не нужно.")
        return []

    names = [
        name
        for name in ("model_initial.cbm", "preproc_initial.json", "metrics_initial.json")
        if (MODELS_DIR / name).exists()
    ]
    if not names:
        raise FileNotFoundError(f"В {MODELS_DIR} нет стартовой версии initial")

    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(MODELS_DIR / name, BOOTSTRAP_DIR / name)
    (BOOTSTRAP_DIR / LATEST_POINTER).write_text("initial\n", encoding="utf-8")
    logger.info("Стартовая версия скопирована в %s", BOOTSTRAP_DIR)
    return names
