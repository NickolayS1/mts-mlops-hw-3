"""Разовая инициализация стенда: витрина, состояние окна, стартовая модель.

Запускается один раз при `docker compose up` и завершается. Все шаги проверяют
свои условия, поэтому повторный запуск контейнера ничего не портит.
"""

import logging
import sys
from pathlib import Path

# Пакет pipeline лежит в корне проекта, в контейнере смонтирован в /app/../pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steps import (  # noqa: E402
    connect,
    ensure_schema,
    ensure_training_state,
    load_csv,
    materialize_window,
    mirror_starter_to_bootstrap,
)
from train_starter import train_starter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_loader")


def main() -> None:
    """Прогоняет шаги инициализации по порядку и закрывает соединение."""
    logger.info("Инициализация стенда начата")
    conn = connect()
    try:
        ensure_schema(conn)
        load_csv(conn)
        ensure_training_state(conn)
        materialize_window(conn)
    finally:
        conn.close()
    train_starter()
    mirror_starter_to_bootstrap()
    logger.info("Инициализация стенда завершена")


if __name__ == "__main__":
    main()
