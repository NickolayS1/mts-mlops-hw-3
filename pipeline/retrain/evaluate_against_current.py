"""Проверка улучшения модели: хуже предыдущей версии — не публиковать."""

import logging

from airflow.exceptions import AirflowSkipException

from pipeline.settings import KEEP_VERSIONS, METRIC_TOLERANCE, MODELS_DIR
from pipeline.train import cleanup_versions

logger = logging.getLogger(__name__)

# Метрика сравнения одна и та же у обеих версий: ROC-AUC на своём свежем месяце.
COMPARED_METRIC = "roc_auc"


def decide(metrics: dict, previous: dict | None, tolerance: float = METRIC_TOLERANCE) -> tuple[bool, str]:
    """Решает, публиковать ли версию, и объясняет решение одной строкой.

    Первая версия публикуется всегда. Дальше кандидат должен быть не хуже
    предыдущего с учётом допуска: допуск нужен, потому что проверочные месяцы
    разные и ROC-AUC колеблется в третьем знаке.
    """
    if previous is None:
        return True, "предыдущей версии с метриками нет, публикуется как первая"

    candidate_score = metrics.get(COMPARED_METRIC)
    previous_score = previous.get(COMPARED_METRIC)
    if candidate_score is None or previous_score is None:
        return True, f"в метриках нет {COMPARED_METRIC}, сравнение пропущено"
    if candidate_score + tolerance >= previous_score:
        return True, (
            f"{COMPARED_METRIC} {candidate_score:.4f} не хуже "
            f"{previous_score:.4f} минус допуск {tolerance}"
        )
    return False, (
        f"{COMPARED_METRIC} {candidate_score:.4f} хуже "
        f"{previous_score:.4f} минус допуск {tolerance}"
    )


def evaluate_against_current(train_info: dict) -> dict:
    """Сравнивает метрики кандидата с метриками активной версии.

    Читает метрики из train_info, файлы не меняет. Если кандидат хуже,
    помечает задачу skipped: публикация и сдвиг указателя месяца не выполняются,
    активной остаётся предыдущая версия, а артефакты кандидата лежат на диске
    для разбора. Повторный запуск снова обучит и снова сравнит.
    """
    promote, reason = decide(train_info["metrics"], train_info.get("previous_metrics"))
    logger.info("Решение по версии %s: %s (%s)", train_info["version"], "публиковать" if promote else "не публиковать", reason)
    if not promote:
        # Отклонённый кандидат не нужен: его файлы удаляются, чтобы повторные
        # запуски на неизменном окне не копили черновики моделей.
        cleanup_versions(
            MODELS_DIR,
            keep_count=KEEP_VERSIONS,
            keep=[train_info.get("current_version")],
            unfinalized=train_info["version"],
        )
        raise AirflowSkipException(
            f"Версия {train_info['version']} не публикуется: {reason}. "
            f"Активной остаётся {train_info.get('current_version')}."
        )
    return {
        **{key: train_info[key] for key in (
            "version",
            "current_version",
            "batch_month",
            "dropped_month",
            "train_months",
            "test_month",
            "metrics",
            "dataset_path",
        )},
        "promote": promote,
        "reason": reason,
    }
