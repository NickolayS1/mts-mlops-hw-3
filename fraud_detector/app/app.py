"""Сервис скоринга: transactions -> модель -> scores.

Подхватывает опубликованную версию модели без перезапуска: указатель
latest.txt перечитывается на каждом сообщении.
"""

import json
import logging
import os
import sys
import time

import pandas as pd
from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka import KafkaError

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Рядом лежат модули сервиса (src) и смонтированный пакет pipeline: без /app
# в sys.path импорт pipeline.features из src/scorer.py не находится.
sys.path.insert(0, os.path.join(APP_DIR, "..", "src"))
sys.path.insert(0, os.path.join(APP_DIR, ".."))

import settings
from scorer import Scorer, ensure_bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_detector")

CONNECT_ATTEMPTS = 30
CONNECT_PAUSE_SECONDS = 2


def connect_consumer() -> Consumer:
    """Ждёт брокер и подписывается на топик транзакций.

    Группа ml-scorer читает новые сообщения с начала топика: без этого первый
    запуск сервиса пропустил бы транзакции, отправленные до его старта.
    """
    config = {
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": settings.CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    last_error = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            consumer = Consumer(config)
            consumer.subscribe([settings.TRANSACTIONS_TOPIC])
            logger.info(
                "Подписка на %s через %s",
                settings.TRANSACTIONS_TOPIC,
                settings.KAFKA_BOOTSTRAP_SERVERS,
            )
            return consumer
        except KafkaException as exc:
            last_error = exc
            logger.info("Kafka ещё не готова (%s/%s): %s", attempt, CONNECT_ATTEMPTS, exc)
            time.sleep(CONNECT_PAUSE_SECONDS)
    raise last_error


def connect_producer() -> Producer:
    """Открывает продюсер результатов.

    JSON кодируется перед отправкой вручную: сериализатор в конфигурации
    librdkafka не поддерживает, а версии обёртки принимают разные ключи.
    """
    return Producer({"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS})


def parse_message(payload: bytes) -> tuple[str, pd.DataFrame]:
    """Разбирает сообщение транзакции в идентификатор и одну строку данных."""
    message = json.loads(payload.decode("utf-8"))
    transaction_id = message["transaction_id"]
    return transaction_id, pd.DataFrame([message["data"]])


def publish(producer: Producer, transaction_id: str, result: dict) -> None:
    """Пишет результат скоринга в топик scores.

    Ключ сообщения — transaction_id: по нему видно, к какой транзакции
    относится результат, даже если потребитель читает топик вразнобой.
    """
    payload = {"transaction_id": transaction_id, **result}
    producer.produce(
        settings.SCORING_TOPIC,
        key=transaction_id.encode("utf-8"),
        value=json.dumps(payload).encode("utf-8"),
    )
    producer.poll(0)


def run() -> None:
    """Бесконечно скорит транзакции и публикует результат в topic scores."""
    logger.info("Старт сервиса скоринга, модель из %s", settings.MODELS_DIR)
    ensure_bootstrap()
    scorer = Scorer()
    consumer = connect_consumer()
    producer = connect_producer()
    scored = 0
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Ошибка Kafka: %s", message.error())
                continue
            try:
                transaction_id, rows = parse_message(message.value())
                result = scorer.score(rows)[0]
                publish(producer, transaction_id, result)
                scored += 1
                if scored % 100 == 0:
                    logger.info(
                        "Отскорено %s транзакций, активная версия %s",
                        scored,
                        scorer.version,
                    )
            except Exception:
                logger.exception("Сообщение не обработано и будет пропущено")
            consumer.commit(message=message, asynchronous=True)
    finally:
        producer.flush()
        consumer.close()


if __name__ == "__main__":
    run()
