"""Разовая проверка схемы Ф2 на train.csv.

Строит признаки теми же функциями, что и DAG, и сверяет их с ожидаемыми:
доли фрода по сегментам, полнота словаря мерчантов, запасная медиана для
незнакомого мерчанта.
Запуск: python scripts/check_features.py [путь к CSV]
Путь можно задать переменной окружения TRAIN_DATA_CSV.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    check_raw_columns,
    fit_preproc,
    load_preproc,
    save_preproc,
    train_matrix,
)


def dataset_path() -> Path:
    """Путь к CSV: аргумент командной строки, затем TRAIN_DATA_CSV."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    value = os.getenv("TRAIN_DATA_CSV")
    if not value:
        raise SystemExit("Укажите путь к train.csv аргументом или в TRAIN_DATA_CSV")
    return Path(value)


def fraud_rate(frame, mask, label="target"):
    part = frame.loc[mask]
    return len(part), float(part[label].mean()) if len(part) else float("nan")


def main():
    started = time.time()
    raw = pd.read_csv(dataset_path())
    print(f"датасет: {raw.shape[0]} строк, {raw.shape[1]} колонок, {time.time() - started:.1f} с")
    print(f"доля фрода: {raw['target'].mean():.4%}")
    print(f"период: {raw['transaction_time'].min()} … {raw['transaction_time'].max()}")
    print(f"колонки: {list(raw.columns)}")

    check_raw_columns(raw)
    print("\ncheck_raw_columns: ок")

    # Словарь учим на первой половине строк: так проверяется и перенос на новые данные.
    half = len(raw) // 2
    train = raw.iloc[:half]
    preproc = fit_preproc(train, version="check")
    features = build_features(raw, preproc, with_target=True)
    matrix, label = train_matrix(features)
    print(f"признаки: {matrix.shape}, метка: {label.shape}, {time.time() - started:.1f} с")

    print(f"\nпорядок колонок: {list(matrix.columns)}")
    print(f"пропуски в признаках: {int(matrix[NUMERIC_FEATURES].isna().sum().sum())}")
    assert list(matrix.columns) == FEATURES, "порядок колонок разошёлся с FEATURES"
    assert matrix[NUMERIC_FEATURES].notna().to_numpy().any(), "числовые признаки пусты"

    print("\n--- доли фрода по сегментам (сверка с замерами) ---")
    for name, mask in (
        ("ночь 22:00-03:59", features["is_night"] == 1),
        ("день 04:00-21:59", features["is_night"] == 0),
        ("сумма > 500", raw["amount"] > 500),
        ("сумма > 200", raw["amount"] > 200),
        ("сумма <= 50", raw["amount"] <= 50),
        ("ночь и сумма > 200", (features["is_night"] == 1) & (raw["amount"] > 200)),
    ):
        rows, rate = fraud_rate(features, mask)
        print(f"  {name:22} {rows:7} строк  {rate:7.3%}")

    print("\n--- сумма к медиане мерчанта ---")
    ratio = features["amount_per_merchant"]
    for name, mask in (
        ("> 5x", ratio > 5),
        ("2-5x", (ratio > 2) & (ratio <= 5)),
        ("0.5-1x", (ratio > 0.5) & (ratio <= 1)),
        ("< 0.5x", ratio <= 0.5),
    ):
        rows, rate = fraud_rate(features, mask)
        print(f"  {name:8} {rows:7} строк  {rate:7.3%}")

    print("\n--- распределение по месяцам ---")
    months = pd.to_datetime(features["transaction_time"]).dt.to_period("M")
    for month, group in features.groupby(months, sort=True):
        print(f"  {month}  {len(group):7} строк  фрод {group['target'].mean():7.3%}")

    print("\n--- словарь препроцессинга ---")
    medians = pd.Series(preproc["amount_median_by_merchant"], dtype="float64")
    print(f"  мерчантов: {len(medians)}")
    print(f"  общая медиана: {preproc['global_amount_median']:.4f}")
    print(f"  разброс медиан: {medians.min():.2f} … {medians.max():.2f}")
    unseen = raw.iloc[half:]
    unseen = unseen.loc[~unseen["merch"].astype(str).isin(medians.index)]
    print(f"  строк второй половины с мерчантом вне словаря: {len(unseen)}")

    # Все мерчанты датасета попадают в словарь, поэтому запасную медиану
    # проверяем на подменённом имени мерчанта.
    fake = raw.head(5).copy()
    fake["merch"] = "fraud_Новый Мерчант"
    fake_features = build_features(fake, preproc)
    expected_ratio = fake["amount"].to_numpy() / preproc["global_amount_median"]
    print(f"  запасная медиана применилась: "
          f"{np.allclose(fake_features['amount_per_merchant'], expected_ratio)}")
    assert np.allclose(fake_features["amount_per_merchant"], expected_ratio)
    assert np.isfinite(fake_features["amount_per_merchant"]).all(), "запасная медиана не подставилась"
    assert preproc["global_amount_median"] > 0, "общая медиана должна быть положительной"
    expected_log = np.log1p(raw.head(5)["amount"].to_numpy())
    print(f"  amount_log совпал: {np.allclose(fake_features['amount_log'], expected_log)}")
    assert np.allclose(fake_features["amount_log"], expected_log), "логарифм суммы посчитан неверно"

    print("\n--- примеры ---")
    sample = raw.head(3)
    sample_features = build_features(sample, preproc)
    for position, (_, row) in enumerate(sample.iterrows()):
        hours = pd.Timestamp(row["transaction_time"]).hour
        print(
            f"  {row['transaction_time']} hour={hours} "
            f"amount={row['amount']} -> "
            f"log={sample_features['amount_log'].iloc[position]:.4f} "
            f"ratio={sample_features['amount_per_merchant'].iloc[position]:.4f} "
            f"night={sample_features['is_night'].iloc[position]} "
            f"region={sample_features['post_code_region'].iloc[position]:.0f}"
        )
    print(f"\nкатегориальные приведены к строке: "
          f"{[str(matrix[c].dtype) for c in CATEGORICAL_FEATURES]}")

    print("\n--- проверка ошибок ---")
    try:
        build_features(raw.drop(columns=["merch"]))
    except ValueError as exc:
        print(f"  без merch: {exc}")
    try:
        train_matrix(build_features(raw.head(5)))
    except ValueError as exc:
        print(f"  train_matrix без target: {exc}")

    print("\n--- контракт результата ---")
    serve = build_features(raw.head(5), preproc)
    print(f"  без метки: колонок {serve.shape[1]}, target внутри "
          f"{'target' in serve.columns}, время внутри {'transaction_time' in serve.columns}")
    assert "target" not in serve.columns, "метка не должна попадать в скоринг"
    assert "transaction_time" in serve.columns, "время нужно для временного split"

    dictionary = Path("dq_reports") / "preproc_check.json"
    save_preproc(preproc, dictionary)
    restored = load_preproc(dictionary)
    print(f"  словарь из JSON: мерчантов {len(restored['amount_median_by_merchant'])}, "
          f"медиана {restored['global_amount_median']:.4f}, совпал {restored == preproc}")
    assert restored == preproc, "словарь не пережил запись и чтение"
    dictionary.unlink()

    print(f"\nвсего {time.time() - started:.1f} с")


if __name__ == "__main__":
    main()
