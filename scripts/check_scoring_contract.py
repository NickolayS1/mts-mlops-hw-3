"""Проверка контракта скоринга: сообщение симулятора -> матрица признаков.

Считает признаки из такого же словаря, какой отправляет transaction_simulator,
и проверяет форму матрицы, порядок колонок и граничные случаи часов и ZIP.
Модель не нужна: признаки и матрица для predict_proba строятся без неё.
Запуск: python scripts/check_scoring_contract.py [путь к CSV]
Путь можно задать переменной окружения TRAIN_DATA_CSV.
"""

import datetime as dt
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features import (  # noqa: E402
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    feature_matrix,
    fit_preproc,
    load_preproc,
    save_preproc,
)


def dataset_path() -> Path:
    """Путь к CSV: аргумент командной строки, затем TRAIN_DATA_CSV."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    value = os.getenv("TRAIN_DATA_CSV")
    if not value:
        raise SystemExit("Укажите путь к train.csv аргументом или в TRAIN_DATA_CSV")
    return Path(value)


def json_safe(value):
    """Копия приведения значений из transaction_simulator, чтобы контракт совпал."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_message(row: pd.Series, moment: dt.datetime) -> dict:
    """Копия сборки сообщения из transaction_simulator."""
    data = {column: json_safe(value) for column, value in row.items()}
    data["transaction_time"] = moment.strftime("%Y-%m-%d %H:%M")
    return {"transaction_id": str(uuid.uuid4()), "data": data}


def main():
    raw = pd.read_csv(dataset_path(), nrows=5000)
    preproc = fit_preproc(raw, version="contract")
    moment = dt.datetime(2026, 3, 4, 1, 30)

    payload = json.loads(json.dumps(build_message(raw.iloc[0], moment)))
    print(f"сообщение симулятора: {len(payload['data'])} полей, "
          f"id {payload['transaction_id'][:8]}…, время {payload['data']['transaction_time']}")

    # Так сообщение разбирает сервис скоринга: pd.DataFrame([data]).
    row = pd.DataFrame([payload["data"]])
    matrix = feature_matrix(build_features(row, preproc))
    print(f"матрица для predict_proba: {matrix.shape}, "
          f"колонки совпали с FEATURES: {list(matrix.columns) == FEATURES}")
    assert list(matrix.columns) == FEATURES, "порядок признаков разошёлся"
    assert matrix.shape == (1, len(FEATURES)), f"ожидалась строка из {len(FEATURES)} признаков"
    assert matrix[NUMERIC_FEATURES].notna().all().all(), "в числовых признаках появились пропуски"
    assert all(isinstance(value, str) for value in matrix.iloc[0][len(NUMERIC_FEATURES):]), \
        "категориальные признаки должны уходить в CatBoost строками"

    print("\n--- граничные случаи времени ---")
    night = dt.datetime(2026, 3, 4, 23, 59)
    morning = dt.datetime(2026, 3, 4, 3, 59)
    day = dt.datetime(2026, 3, 4, 4, 0)
    saturday = dt.datetime(2026, 3, 7, 12, 0)
    for label, when, expected_night, expected_hour in (
        ("23:59", night, 1, 23),
        ("03:59", morning, 1, 3),
        ("04:00", day, 0, 4),
        ("суббота 12:00", saturday, 0, 12),
    ):
        message = build_message(raw.iloc[0], when)
        features = build_features(pd.DataFrame([message["data"]]), preproc)
        night_flag = int(features["is_night"].iloc[0])
        hour = int(features["hour"].iloc[0])
        weekend = int(features["is_weekend"].iloc[0])
        print(f"  {label:15} hour={hour:2} night={night_flag} weekend={weekend}")
        assert hour == expected_hour, f"{label}: час разобран неверно"
        assert night_flag == expected_night, f"{label}: граница ночи определена неверно"
    assert int(build_features(pd.DataFrame([build_message(raw.iloc[0], saturday)["data"]]),
                              preproc)["is_weekend"].iloc[0]) == 1

    print("\n--- граничные случаи ZIP и суммы ---")
    sample = raw.iloc[0].to_dict()
    cases = [
        ("ZIP 5 знаков", {"post_code": 97033}, 9.0),
        ("ZIP строкой", {"post_code": "01234"}, 0.0),
        ("ZIP пустой", {"post_code": None}, -1.0),
        ("ZIP мусор", {"post_code": "abc"}, -1.0),
    ]
    for label, changes, expected_region in cases:
        message = build_message(pd.Series({**sample, **changes}), moment)
        features = build_features(pd.DataFrame([message["data"]]), preproc)
        region = float(features["post_code_region"].iloc[0])
        print(f"  {label:14} region={region:5.1f}")
        assert region == expected_region, f"{label}: регион извлечён неверно"

    for label, amount in (("сумма 0", 0.0), ("сумма отрицательная", -5.0)):
        message = build_message(pd.Series({**sample, "amount": amount}), moment)
        features = build_features(pd.DataFrame([message["data"]]), preproc)
        log_amount = float(features["amount_log"].iloc[0])
        print(f"  {label:20} amount_log={log_amount:.3f}")
        assert log_amount >= 0, f"{label}: логарифм ушёл в минус"

    print("\n--- пара модель+словарь и указатель версии ---")
    with tempfile.TemporaryDirectory() as directory:
        models = Path(directory)
        (models / "model_v1.cbm").write_bytes(b"stub")
        pointer = models / "latest.txt"

        # Публикация: сначала пара, потом указатель. Указателя ещё нет.
        assert not pointer.exists()
        save_preproc(preproc, models / "preproc_v1.json")
        pointer.write_text("v1\n", encoding="utf-8")
        version = pointer.read_text(encoding="utf-8").strip()
        print(f"  указатель указывает на {version}, модель есть: "
              f"{(models / f'model_{version}.cbm').exists()}, "
              f"словарь есть: {(models / f'preproc_{version}.json').exists()}")
        assert (models / f"model_{version}.cbm").exists()
        assert (models / f"preproc_{version}.json").exists()
        assert load_preproc(models / f"preproc_{version}.json") == preproc, \
            "словарь версии не совпал с записанным"

        # Публикация прервана: модель новой версии есть, словаря нет.
        (models / "model_v2.cbm").write_bytes(b"stub")
        broken = "v2"
        print(f"  прерванная публикация {broken}: словарь отсутствует "
              f"{not (models / f'preproc_{broken}.json').exists()}")
        assert not (models / f"preproc_{broken}.json").exists(), \
            "тест на неполную пару потерял смысл"
        try:
            load_preproc(models / f"preproc_{broken}.json")
            raise AssertionError("чтение отсутствующего словаря должно падать")
        except FileNotFoundError:
            print("  неполная пара распознаётся: load_preproc падает на отсутствующем файле")

    print("\nконтракт скоринга проверен")


if __name__ == "__main__":
    main()
