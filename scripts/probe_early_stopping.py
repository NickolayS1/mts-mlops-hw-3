"""Сравнение терпения early stopping на первом окне датасета.

2 дерева у стартовой модели выглядят подозрительно: обучение обрывается, пока
метрика ещё растёт. Проверка показывает, сколько деревьев получается при разном
early_stopping_rounds на одном и том же сплите.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.features import CATEGORICAL_FEATURES, build_features, fit_preproc, train_matrix  # noqa: E402


def dataset_path() -> Path:
    """Путь к CSV: аргумент командной строки, затем TRAIN_DATA_CSV."""
    parser = argparse.ArgumentParser(description="Сравнение терпения early stopping")
    parser.add_argument("csv", nargs="?", default=os.getenv("TRAIN_DATA_CSV"))
    args = parser.parse_args()
    if not args.csv:
        raise SystemExit("Укажите путь к train.csv аргументом или в TRAIN_DATA_CSV")
    return Path(args.csv)


def main() -> None:
    frame = pd.read_csv(dataset_path(), parse_dates=["transaction_time"])
    frame = frame.loc[frame["transaction_time"] < pd.Timestamp("2019-04-01")]
    periods = frame["transaction_time"].dt.to_period("M")
    latest = periods.max()
    train_part = frame.loc[periods < latest]
    test_part = frame.loc[periods == latest]
    print(f"train {len(train_part)} строк (месяцы {sorted(periods.loc[periods < latest].unique().astype(str))}), "
          f"test {len(test_part)} строк ({latest})")

    preproc = fit_preproc(train_part, version="probe")
    x_train, y_train = train_matrix(build_features(train_part, preproc, with_target=True))
    x_test, y_test = train_matrix(build_features(test_part, preproc, with_target=True))

    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    for patience in (50, 150, 300):
        started = time.time()
        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            early_stopping_rounds=patience,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            x_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES,
            eval_set=(x_test, y_test),
            use_best_model=True,
        )
        probabilities = model.predict_proba(x_test)[:, 1]
        print(f"  early_stopping_rounds={patience:4}: деревьев {model.tree_count_:4}, "
              f"ROC-AUC={roc_auc_score(y_test, probabilities):.4f}, "
              f"{time.time() - started:.1f} с")


if __name__ == "__main__":
    main()
