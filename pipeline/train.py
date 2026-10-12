"""Обучение CatBoost и метрики на свежем месяце — одно место для всех версий.

Стартовую модель в data_loader и переобучение в DAG тренирует эта функция,
поэтому схема признаков, параметры и способ подсчёта метрик не могут разойтись.
"""

import json
import logging
import time
from datetime import datetime, timezone

from .features import (
    CATEGORICAL_FEATURES,
    build_features,
    fit_preproc,
    metrics_path,
    save_preproc,
    train_matrix,
)

logger = logging.getLogger(__name__)

CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    # Терпение проверено на первом окне: при 50 обучение обрывалось на 63 деревьях
    # с ROC-AUC 0.9771, при 150 — 273 дерева и 0.9821. Больше 150 не даёт прироста.
    "early_stopping_rounds": 150,
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 50,
}

THRESHOLD = 0.5


def new_version() -> str:
    """Версия по времени в UTC. Различима для версий, опубликованных в одну секунду."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def read_metrics(version: str, models_dir) -> dict | None:
    """Читает метрики версии. None — если версия старше появления файлов метрик."""
    path = metrics_path(version, models_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def train_model(train_frame, test_frame, models_dir, version: str) -> tuple[dict, dict]:
    """Обучает модель на train и считает метрики на свежем месяце.

    Словарь Ф2 считается только по train_frame: проверочный месяц не должен
    влиять на медианы, иначе метрика окажется завышенной.
    Возвращает обученную модель и словарь метрик. Файлы не пишет.
    """
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    preproc = fit_preproc(train_frame, version=version)
    x_train, y_train = train_matrix(build_features(train_frame, preproc, with_target=True))
    x_test, y_test = train_matrix(build_features(test_frame, preproc, with_target=True))
    if y_test.nunique() < 2:
        raise RuntimeError("В проверочном месяце одна метка, метрики не определены")

    started = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        x_train,
        y_train,
        # Категориальные признаки по именам: порядок колонок матрицы фиксируется
        # в модели, и скоринг обязан подать те же имена в том же порядке.
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_test, y_test),
        use_best_model=True,
    )
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "version": version,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_months": sorted(
            train_frame["transaction_time"].dt.to_period("M").unique().astype(str)
        ),
        "test_month": str(test_frame["transaction_time"].dt.to_period("M").max()),
        "trees": int(model.tree_count_),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, (probabilities > THRESHOLD).astype(int), zero_division=0)),
    }
    logger.info(
        "Модель %s: ROC-AUC=%.4f, F1=%.4f, деревьев %s, %.1f с",
        version,
        metrics["roc_auc"],
        metrics["f1"],
        metrics["trees"],
        time.time() - started,
    )
    return model, {"metrics": metrics, "preproc": preproc}


def write_pair(model, preproc: dict, metrics: dict, models_dir, staging: bool = False) -> None:
    """Пишет пару модель+словарь и файл метрик для указанной версии.

    При staging модель получает префикс .staging: кандидат виден на диске для
    разбора, но его файл не может быть выбран указателем latest.txt по имени.
    Словарь и метрики пишутся окончательными именами — они относятся к версии,
    а не к активной модели. Порядок записи фиксирован: словарь, модель, метрики.
    """
    from pathlib import Path

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = metrics["version"]
    prefix = ".staging_" if staging else ""
    model.save_model(str(models_dir / f"{prefix}model_{version}.cbm"))
    save_preproc(preproc, models_dir / f"preproc_{version}.json")
    metrics_path(version, models_dir).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Записана пара версии %s в %s", version, models_dir)


def remove_incomplete_pairs(models_dir) -> list[str]:
    """Удаляет модели, у которых нет словаря Ф2.

    Пара публикуется целиком, поэтому модель без словаря не загружается
    скорингом и восстановлению не подлежит. Такие файлы появляются, если
    публикация была прервана или артефакты удалили вручную.
    """
    from pathlib import Path

    models_dir = Path(models_dir)
    removed = []
    for model_file in sorted(models_dir.glob("model_*.cbm")):
        version = model_file.stem[len("model_") :]
        if (models_dir / f"preproc_{version}.json").exists():
            continue
        model_file.unlink()
        for name in (f"metrics_{version}.json", f"preproc_{version}.json"):
            target = models_dir / name
            if target.exists():
                target.unlink()
        removed.append(version)
        logger.warning("Удалена модель %s без словаря Ф2", version)
    return removed


def cleanup_versions(
    models_dir,
    keep_count: int,
    keep: list[str | None] | None = None,
    unfinalized: str | None = None,
) -> list[str]:
    """Удаляет артефакты версий, которые точно никому не нужны.

    Удаляются только версии, названные явно: непубликуемый кандидат
    (unfinalized) и, если пар больше keep_count, самые старые из оставшихся.
    Активная версия и версии из keep не удаляются никогда. Если активной
    версии нет (например, пропал latest.txt), устаревшие версии не удаляются:
    молчаливое удаление по пустому набору уже приводило к потере
    опубликованной модели. Возвращает список удалённых версий.
    """
    from pathlib import Path

    models_dir = Path(models_dir)
    protected = {version for version in (keep or []) if version}
    if unfinalized and unfinalized in protected:
        raise RuntimeError(f"Версия {unfinalized} защищена, удалять её нельзя")
    removed = []

    if unfinalized:
        for name in (
            f".staging_model_{unfinalized}.cbm",
            f"preproc_{unfinalized}.json",
            f"metrics_{unfinalized}.json",
        ):
            target = models_dir / name
            if target.exists():
                target.unlink()
        removed.append(unfinalized)
        logger.info("Артефакты непубликуемой версии %s удалены", unfinalized)

    if not protected:
        logger.warning("Активная версия неизвестна, устаревшие версии не удаляются.")
    else:
        spare = len(protected)
        versions = sorted(
            (path.stem[len("model_") :] for path in models_dir.glob("model_*.cbm")),
            reverse=True,
        )
        for version in versions:
            if version in protected or version == unfinalized:
                continue
            if spare < keep_count:
                spare += 1
                continue
            for name in (f"model_{version}.cbm", f"preproc_{version}.json", f"metrics_{version}.json"):
                target = models_dir / name
                if target.exists():
                    target.unlink()
            removed.append(version)
            logger.info("Удалена устаревшая версия %s", version)

    # Неполные пары убираются всегда: модель без словаря не загружается,
    # и её наличие не зависит от того, известна ли активная версия.
    removed.extend(remove_incomplete_pairs(models_dir))
    return removed
