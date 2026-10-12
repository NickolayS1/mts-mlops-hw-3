"""Обучение стартовой модели на первом окне датасета.

Проверяет, что опубликованная пара грузится, что порядок признаков модели
совпадает с порядком, который подаёт скоринг, и что скалярное сообщение
из Kafka превращается в один скор. Нужен CatBoost, поэтому запускается
в .venv, а не системным Python.
Запуск: .venv/Scripts/python.exe scripts/train_initial.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_loader"))

WINDOW_END = pd.Timestamp("2019-03-31 23:59")


def dataset_path() -> Path:
    """Путь к CSV: аргумент командной строки, затем TRAIN_DATA_CSV."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    value = os.getenv("TRAIN_DATA_CSV")
    if not value:
        raise SystemExit("Укажите путь к train.csv аргументом или в TRAIN_DATA_CSV")
    return Path(value)


def main() -> None:
    training_dir = Path(tempfile.mkdtemp(prefix="dsh_training_"))
    models_dir = Path(tempfile.mkdtemp(prefix="dsh_models_"))

    import pipeline.settings as settings

    settings.CURRENT_DATASET = training_dir / "current.parquet"
    settings.TRAINING_DIR = training_dir
    settings.MODELS_DIR = models_dir

    import steps
    import train_starter

    steps.CURRENT_DATASET = settings.CURRENT_DATASET
    train_starter.MODELS_DIR = models_dir
    train_starter.CURRENT_DATASET = settings.CURRENT_DATASET

    started = pd.Timestamp.now()
    window = pd.read_csv(dataset_path(), parse_dates=["transaction_time"])
    window = window.loc[window["transaction_time"] <= WINDOW_END]
    window.to_parquet(settings.CURRENT_DATASET, index=False)
    print(f"окно обучения: {len(window)} строк, "
          f"{window['transaction_time'].min()} … {window['transaction_time'].max()}, "
          f"{(pd.Timestamp.now() - started).total_seconds():.1f} с подготовки")
    print(f"месяцы: {sorted(window['transaction_time'].dt.to_period('M').unique().astype(str))}")

    version = train_starter.train_starter()
    print(f"\nверсия: {version}")
    print(f"файлы в каталоге моделей: {sorted(path.name for path in models_dir.iterdir())}")

    model_file = models_dir / "model_initial.cbm"
    preproc_file = models_dir / "preproc_initial.json"
    pointer = models_dir / "latest.txt"
    assert model_file.exists() and preproc_file.exists() and pointer.exists(), "пара не опубликована"
    assert pointer.read_text(encoding="utf-8").strip() == version

    from catboost import CatBoostClassifier
    from pipeline.features import CATEGORICAL_FEATURES, FEATURES, load_preproc

    model = CatBoostClassifier()
    model.load_model(str(model_file))
    declared = list(model.feature_names_)
    print(f"\nмодель объявляет признаки: {declared}")
    print(f"совпали с FEATURES: {declared == FEATURES}")
    assert declared == FEATURES, "порядок признаков в модели разошёлся со схемой Ф2"
    cat_indices = model.get_cat_feature_indices()
    cat_names = [declared[index] for index in cat_indices]
    print(f"категориальные в модели: {cat_names}")
    assert cat_names == CATEGORICAL_FEATURES, "набор категориальных признаков разошёлся"

    preproc = load_preproc(preproc_file)
    print(f"словарь: мерчантов {len(preproc['amount_median_by_merchant'])}, "
          f"общая медиана {preproc['global_amount_median']:.4f}")

    # Путь скоринга: то же, что делает fraud_detector/src/scorer.py.
    sys.path.insert(0, str(ROOT / "fraud_detector" / "src"))
    from pipeline.features import build_features, feature_matrix

    sample = window.sample(n=5, random_state=1)
    matrix = feature_matrix(build_features(sample, preproc))
    probabilities = model.predict_proba(matrix)[:, 1]
    print("\nскоринг пробных сообщений:")
    for probability, label in zip(probabilities, sample["target"]):
        print(f"  score={probability:.4f}  fraud_flag={int(probability > 0.5)}  метка={int(label)}")
    assert len(probabilities) == 5

    metrics = (models_dir / "metrics_initial.json").read_text(encoding="utf-8")
    print(f"\nmetrics_initial.json:\n{metrics}")


if __name__ == "__main__":
    main()
