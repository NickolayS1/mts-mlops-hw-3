"""Схема Ф2 в контейнере скоринга.

Реализация живёт в pipeline/features.py и монтируется в образ, чтобы обучение
в Airflow и инференс в сервисе считали признаки одним кодом. Здесь только
точка входа: сервису нужен один модуль, а не весь пакет pipeline.
"""

from pipeline.features import (  # noqa: F401
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_features,
    feature_matrix,
    load_preproc,
    save_preproc,
    train_matrix,
)
