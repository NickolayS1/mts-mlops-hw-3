"""Схема признаков Ф2, общая для обучения в Airflow и для скоринга.

Модуль лежит в pipeline/, а контейнер скоринга монтирует его и импортирует
отсюда же: одна реализация на обучение и на инференс, расхождение схемы
невозможно. Словарь препроцессинга строится только по обучающей части окна,
поэтому в скоринге не появляется новых преобразований.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "amount_log",
    "amount_per_merchant",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_night",
    "is_weekend",
    "post_code_region",
]

# Имена категориальных колонок взяты из train.csv: cat_id — категория
# мерчанта, one_city — округ. CatBoost принимает их строками.
CATEGORICAL_FEATURES = [
    "merch",
    "cat_id",
    "gender",
    "one_city",
    "us_state",
    "jobs",
    "post_code",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL = "target"
TIME_COLUMN = "transaction_time"

RAW_COLUMNS = [
    TIME_COLUMN,
    "merch",
    "cat_id",
    "amount",
    "name_1",
    "name_2",
    "gender",
    "street",
    "one_city",
    "us_state",
    "post_code",
    "lat",
    "lon",
    "population_city",
    "jobs",
    "merchant_lat",
    "merchant_lon",
    "target",
]

# Отбрасываются: идентификаторы человека, координаты и производные от них,
# население города, исходная сумма (заменена логарифмом) и сама метка.
DROPPED_COLUMNS = [
    "name_1",
    "name_2",
    "street",
    "lat",
    "lon",
    "merchant_lat",
    "merchant_lon",
    "population_city",
    "amount",
    "target",
]

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 4


def fit_preproc(train_frame: pd.DataFrame, version: str) -> dict:
    """Считает словарь преобразований по обучающей части окна.

    Медиана суммы по каждому мерчанту нужна признаку amount_per_merchant.
    Общая медиана остаётся запасной для мерчантов, которых не было в обучении.
    Исходный frame не меняется. Возвращает словарь для save_preproc.
    """
    amounts = pd.to_numeric(train_frame["amount"], errors="coerce")
    medians = (
        pd.DataFrame({"merch": train_frame["merch"].astype(str), "amount": amounts})
        .dropna(subset=["amount"])
        .groupby("merch")["amount"]
        .median()
    )
    preproc = {
        "version": version,
        "global_amount_median": float(amounts.median()),
        "amount_median_by_merchant": {
            merch: float(value) for merch, value in medians.items()
        },
    }
    logger.info(
        "Словарь Ф2 %s: медиан по мерчантам %s, общая медиана %.4f",
        version,
        len(preproc["amount_median_by_merchant"]),
        preproc["global_amount_median"],
    )
    return preproc


def save_preproc(preproc: dict, path) -> None:
    """Пишет словарь Ф2 в JSON. sort_keys делает файл побайтово воспроизводимым.

    Повторная запись того же словаря по тому же пути ничего не ломает: файл
    либо остаётся прежним, либо полностью заменяется новой версией.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(preproc, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Словарь Ф2 записан: %s", target)


def load_preproc(path) -> dict:
    """Читает словарь Ф2, сохранённый save_preproc.

    Скоринг вызывает функцию на каждой перезагрузке модели, поэтому словарь
    читается заново вместе с моделью из указателя latest.txt.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_path(version: str, models_dir) -> Path:
    """Путь к файлу метрик версии.

    Метрики соседствуют с парой модель+словарь: по ним следующая версия
    сравнивается с текущей, поэтому они публикуются вместе с моделью.
    """
    return Path(models_dir) / f"metrics_{version}.json"


def check_raw_columns(frame: pd.DataFrame) -> None:
    """Проверяет, что во входных данных есть все колонки схемы Ф2."""
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"В данных нет колонок схемы Ф2: {missing}")


def build_features(
    frame: pd.DataFrame,
    preproc: dict | None = None,
    with_target: bool = False,
) -> pd.DataFrame:
    """Строит признаки Ф2 из сырых строк витрины.

    Возвращает признаки, время транзакции и, при with_target, метку; порядок
    колонок фиксирован списком FEATURES. transaction_time остаётся, чтобы
    задачи дальше делили окно по времени, а не по позиции строк.
    preproc нужен для amount_per_merchant; без него берётся общая медиана
    и признак равен логарифму суммы.
    """
    check_raw_columns(frame)

    features = pd.DataFrame(index=frame.index)
    moments = pd.to_datetime(frame[TIME_COLUMN])
    hours = moments.dt.hour
    weekdays = moments.dt.dayofweek

    amounts = pd.to_numeric(frame["amount"], errors="coerce")
    medians = pd.Series(np.nan, index=frame.index, dtype="float64")
    if preproc:
        merchants = frame["merch"].astype(str)
        by_merchant = preproc["amount_median_by_merchant"]
        medians = merchants.map(by_merchant).astype("float64")
        # Новый мерчант и нулевая медиана получают общую медиану обучения.
        medians = medians.where(medians > 0, float(preproc["global_amount_median"]))
    features["amount_log"] = np.log1p(amounts.clip(lower=0))
    features["amount_per_merchant"] = amounts / medians

    features["hour"] = hours.astype("int64")
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["day_of_week"] = weekdays.astype("int64")
    features["is_night"] = ((hours >= NIGHT_START_HOUR) | (hours < NIGHT_END_HOUR)).astype("int64")
    features["is_weekend"] = (weekdays >= 5).astype("int64")

    # ZIP вида "97033": регион — первая цифра, остальное не различает скор.
    digits = frame["post_code"].astype("string").str.strip().str[0]
    features["post_code_region"] = pd.to_numeric(digits, errors="coerce").fillna(-1).astype("float64")

    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].astype("string").fillna("")

    result = features.loc[:, FEATURES].copy()
    result[TIME_COLUMN] = moments
    if with_target:
        result[LABEL] = frame[LABEL].astype("int64")
    return result


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Приводит колонки к типам, которые ждёт CatBoost, и отдаёт матрицу.

    Категориальные признаки остаются строками, числовые — числами. Метка не
    нужна: функция вызывается и в обучении, и в скоринге, где метки нет.
    """
    matrix = features.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        matrix[column] = pd.to_numeric(matrix[column])
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype(str)
    return matrix


def train_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет метку от признаков для обучения.

    Метка в матрицу не попадает, поэтому порядок колонок совпадает с тем,
    который модель увидит в скоринге.
    """
    if LABEL not in features.columns:
        raise ValueError(f"В признаках нет колонки {LABEL}")
    return feature_matrix(features), features[LABEL].astype("int64")
