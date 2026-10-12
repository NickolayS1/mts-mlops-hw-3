"""Загрузка пары модель+словарь и горячая подмена версии без перезапуска.

Указатель latest.txt читается перед каждым сообщением. Версия публикуется
только вместе с парой файлов, поэтому скоринг никогда не видит модель без
словаря: сначала появляется пара, потом указатель.
"""

import logging
import shutil
import threading
import time
from pathlib import Path

from pipeline.features import build_features, feature_matrix, load_preproc
from settings import (
    BOOTSTRAP_ATTEMPTS,
    BOOTSTRAP_DIR,
    BOOTSTRAP_PAUSE_SECONDS,
    FRAUD_THRESHOLD,
    LATEST_POINTER,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def model_path(version: str) -> Path:
    """Путь к бинарнику модели указанной версии."""
    return MODELS_DIR / f"model_{version}.cbm"


def preproc_path(version: str) -> Path:
    """Путь к словарю Ф2 указанной версии."""
    return MODELS_DIR / f"preproc_{version}.json"


def ensure_bootstrap() -> None:
    """Ждёт начальную пару модель+словарь и копирует её, пока указателя нет.

    Сервис и обучение стартовой модели запускаются из compose параллельно,
    поэтому каталог моделей может быть ещё пуст: ожидание снимает гонку.
    Если latest.txt уже есть, каталог моделей не трогается: повторный запуск
    контейнера не откатывает опубликованную версию.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pointer = MODELS_DIR / LATEST_POINTER
    for attempt in range(1, BOOTSTRAP_ATTEMPTS + 1):
        if pointer.exists() and pointer.read_text(encoding="utf-8").strip():
            return
        bundled = BOOTSTRAP_DIR / LATEST_POINTER
        if bundled.exists() and bundled.read_text(encoding="utf-8").strip():
            break
        logger.info(
            "Начальная модель ещё не готова (%s/%s)", attempt, BOOTSTRAP_ATTEMPTS
        )
        time.sleep(BOOTSTRAP_PAUSE_SECONDS)
    else:
        raise FileNotFoundError(
            f"Не дождались начальной модели ни в {MODELS_DIR}, ни в {BOOTSTRAP_DIR}"
        )

    version = (BOOTSTRAP_DIR / LATEST_POINTER).read_text(encoding="utf-8").strip()
    for name in (f"model_{version}.cbm", f"preproc_{version}.json", f"metrics_{version}.json"):
        source = BOOTSTRAP_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Нет файла начальной пары: {source}")
        shutil.copyfile(source, MODELS_DIR / name)
        logger.info("Начальный файл скопирован в %s", MODELS_DIR / name)
    # Указатель последним: сначала должна появиться вся пара целиком.
    pointer.write_text(version + "\n", encoding="utf-8")


def read_latest_version() -> str:
    """Читает версию из указателя, который переписывает DAG переобучения."""
    pointer = MODELS_DIR / LATEST_POINTER
    if not pointer.exists():
        raise FileNotFoundError(f"Нет указателя модели: {pointer}")
    version = pointer.read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError(f"Пустой указатель модели: {pointer}")
    return version


def load_pair(version: str):
    """Загружает модель и её словарь одной версии.

    Обе части обязательны: отсутствие словаря означает, что публикация была
    прервана между копированием модели и словаря.
    """
    from catboost import CatBoostClassifier

    path = model_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла модели версии {version}: {path}")
    dictionary = preproc_path(version)
    if not dictionary.exists():
        raise FileNotFoundError(f"Нет словаря Ф2 версии {version}: {dictionary}")

    model = CatBoostClassifier()
    model.load_model(str(path))
    logger.info("Загружена пара версии %s из %s", version, MODELS_DIR)
    return model, load_preproc(dictionary)


class Scorer:
    """Держит текущую пару модель+словарь и подменяет её по указателю.

    Перезагрузка входит в обработку сообщения, поэтому исключение из load_pair
    не убивает сервис: следующее сообщение повторит попытку по тому же указателю.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = None
        self._model = None
        self._preproc = None

    @property
    def version(self) -> str | None:
        """Версия загруженной пары, нужна для логов."""
        return self._version

    def _current(self):
        """Возвращает пару для версии из указателя, перечитывая её при смене.

        Незагружаемая версия не роняет сервис: ошибка логируется, а скоринг
        продолжает работать на последней исправной паре. Без этого пропавший
        или недописанный словарь останавливал бы весь поток транзакций.
        """
        version = read_latest_version()
        if self._model is not None and version == self._version:
            return self._model, self._preproc, self._version
        try:
            model, preproc = load_pair(version)
        except (OSError, ValueError) as exc:
            if self._model is None:
                raise
            logger.error(
                "Версия %s не загружена (%s), продолжаю на версии %s",
                version,
                exc,
                self._version,
            )
            return self._model, self._preproc, self._version
        self._model, self._preproc = model, preproc
        self._version = version
        logger.info("Активна версия модели %s", version)
        return self._model, self._preproc, self._version

    def score(self, rows) -> list[dict]:
        """Считает вероятность фрода для сырых строк транзакций.

        Возвращает по словарю на строку: скор, метку по порогу и версию модели,
        которой скор посчитан. Витрину и Kafka не трогает.
        """
        with self._lock:
            model, preproc, version = self._current()
            matrix = feature_matrix(build_features(rows, preproc))
            probabilities = model.predict_proba(matrix)[:, 1]
        return [
            {
                "score": float(probability),
                "fraud_flag": int(probability > FRAUD_THRESHOLD),
                "model_version": version,
            }
            for probability in probabilities
        ]
