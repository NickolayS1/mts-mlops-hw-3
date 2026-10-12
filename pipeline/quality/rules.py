"""Правила Data Quality для окна обучения.

Каждое правило возвращает наблюдение и вердикт, ничего не меняет и не зависит
от других правил, поэтому отчет собирается по всем правилам сразу.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Колонки, без которых строку нельзя ни обучить, ни отскорить.
REQUIRED_COLUMNS = ["transaction_time", "merch", "cat_id", "amount", "target"]
# Категориальные признаки схемы Ф2.
CATEGORICAL_COLUMNS = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs", "post_code"]

MAX_MISSING_SHARE = 0.0
FRAUD_RATE_MIN = 0.0005
FRAUD_RATE_MAX = 0.05
MAX_DUPLICATE_SHARE = 0.001
MAX_NEW_MERCHANT_SHARE = 0.05
# Признаки с малым числом значений должны быть исчерпывающе известны: новое
# значение такого признака означает, что изменился справочник.
MAX_NEW_LOW_CARDINALITY_VALUES = 0
LOW_CARDINALITY_LIMIT = 20
# Для городов и ZIP новые значения появляются естественно, поэтому порог задан
# долей строк: массовое появление неизвестных значений означает смену источника.
MAX_NEW_HIGH_CARDINALITY_SHARE = 0.05


def result(rule: str, passed: bool, observed, threshold, message: str) -> dict:
    """Собирает результат одного правила в едином виде для отчета."""
    return {
        "rule": rule,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "message": message,
    }


def check_required_columns(frame: pd.DataFrame) -> list[dict]:
    """Все обязательные колонки присутствуют в наборе."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    return [
        result(
            "required_columns",
            not missing,
            missing,
            "все обязательные колонки на месте",
            "нет колонок: " + ", ".join(missing) if missing else "набор колонок полный",
        )
    ]


def check_missing_values(frame: pd.DataFrame) -> list[dict]:
    """Доля пропусков в обязательных колонках не превышает MAX_MISSING_SHARE."""
    checks = []
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            continue
        share = float(frame[column].isna().mean())
        checks.append(
            result(
                f"missing_values.{column}",
                share <= MAX_MISSING_SHARE,
                round(share, 6),
                f"≤ {MAX_MISSING_SHARE}",
                f"пропусков {share:.4%}",
            )
        )
    return checks


def check_time_range(frame: pd.DataFrame, expected_start, expected_end) -> list[dict]:
    """Все отметки времени лежат внутри объявленного периода наблюдений."""
    moments = pd.to_datetime(frame["transaction_time"])
    start, end = pd.Timestamp(expected_start), pd.Timestamp(expected_end)
    outside = int(((moments < start) | (moments > end)).sum())
    return [
        result(
            "time_range",
            outside == 0,
            outside,
            f"0 наблюдений вне {start.date()} … {end.date()}",
            f"вне периода: {outside}",
        )
    ]


def check_target_values(frame: pd.DataFrame) -> list[dict]:
    """Метка принимает только значения 0 и 1."""
    values = sorted(pd.unique(frame["target"].dropna()).tolist())
    unexpected = [value for value in values if int(value) not in (0, 1)]
    return [
        result(
            "target_values",
            not unexpected,
            values,
            "только 0 и 1",
            f"встреченные значения: {values}",
        )
    ]


def check_fraud_rate(frame: pd.DataFrame) -> list[dict]:
    """Доля фрода попадает в разумный диапазон: иначе метки или выборка сломаны."""
    rate = float(pd.to_numeric(frame["target"], errors="coerce").mean())
    passed = FRAUD_RATE_MIN <= rate <= FRAUD_RATE_MAX
    return [
        result(
            "fraud_rate",
            passed,
            round(rate, 6),
            f"{FRAUD_RATE_MIN} … {FRAUD_RATE_MAX}",
            f"доля фрода {rate:.4%}",
        )
    ]


def check_positive_amount(frame: pd.DataFrame) -> list[dict]:
    """Сумма транзакции строго положительна."""
    amounts = pd.to_numeric(frame["amount"], errors="coerce")
    bad = int((amounts <= 0).sum())
    return [
        result(
            "positive_amount",
            bad == 0,
            bad,
            "0 неположительных сумм",
            f"неположительных сумм: {bad}",
        )
    ]


def check_duplicates(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> list[dict]:
    """Доля дублей по паре время+мерчант+сумма ниже MAX_DUPLICATE_SHARE.

    Дубль означает, что один и тот же платёж попал в витрину дважды: COPY в
    витрину не проверяет повторы, поэтому проверка нужна после дозаписи месяца.
    """
    keys = ["transaction_time", "merch", "amount"]
    available = [column for column in keys if column in frame.columns]
    duplicates = int(frame.duplicated(subset=available).sum())
    share = duplicates / len(frame) if len(frame) else 0.0
    return [
        result(
            "duplicates",
            share <= MAX_DUPLICATE_SHARE,
            round(share, 6),
            f"≤ {MAX_DUPLICATE_SHARE}",
            f"дублей {duplicates} ({share:.4%})",
        )
    ]


def check_known_categories(frame: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    """Категориальные значения входят в список, известный по референсу.

    Референс — более ранние месяцы того же окна. Для признаков с малым числом
    значений требуется полное совпадение справочника; для городов и ZIP
    ограничивается доля строк с новым значением, иначе правило срабатывало бы
    на каждом новом месяце: по замерам новых городов 0.10 %, новых ZIP 0.13 %.
    """
    checks = []
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns or column not in reference.columns:
            continue
        known = set(reference[column].astype("string").dropna().unique())
        actual = frame[column].astype("string")
        unknown = sorted(set(actual.dropna().unique()) - known)
        if len(known) <= LOW_CARDINALITY_LIMIT:
            checks.append(
                result(
                    f"known_categories.{column}",
                    len(unknown) <= MAX_NEW_LOW_CARDINALITY_VALUES,
                    len(unknown),
                    f"≤ {MAX_NEW_LOW_CARDINALITY_VALUES} новых значений при {len(known)} известных",
                    f"новых значений {len(unknown)}: {unknown[:5]}",
                )
            )
            continue
        share = float((~actual.isin(known)).mean())
        checks.append(
            result(
                f"known_categories.{column}",
                share <= MAX_NEW_HIGH_CARDINALITY_SHARE,
                round(share, 6),
                f"≤ {MAX_NEW_HIGH_CARDINALITY_SHARE} строк с новым значением",
                f"известно {len(known)} значений, новых {len(unknown)}, доля строк {share:.4%}",
            )
        )
    return checks


def check_new_merchants(frame: pd.DataFrame, reference: pd.DataFrame) -> list[dict]:
    """Доля строк с мерчантами вне референса ниже MAX_NEW_MERCHANT_SHARE."""
    known = set(reference["merch"].astype("string").dropna().unique())
    actual = frame["merch"].astype("string")
    share = float((~actual.isin(known)).mean())
    return [
        result(
            "new_merchants",
            share <= MAX_NEW_MERCHANT_SHARE,
            round(share, 6),
            f"≤ {MAX_NEW_MERCHANT_SHARE}",
            f"строк с новыми мерчантами {share:.4%}",
        )
    ]


def run_all(frame: pd.DataFrame, reference: pd.DataFrame, expected_start, expected_end) -> list[dict]:
    """Прогоняет все правила и возвращает наблюдения по каждому."""
    checks = []
    checks += check_required_columns(frame)
    if checks and not checks[0]["passed"]:
        logger.warning("Часть колонок отсутствует, остальные правила пропущены.")
        return checks
    checks += check_missing_values(frame)
    checks += check_time_range(frame, expected_start, expected_end)
    checks += check_target_values(frame)
    checks += check_fraud_rate(frame)
    checks += check_positive_amount(frame)
    checks += check_duplicates(frame)
    checks += check_known_categories(frame, reference)
    checks += check_new_merchants(frame, reference)
    return checks
