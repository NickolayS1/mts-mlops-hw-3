"""Запись результатов скоринга из Kafka в таблицу scores.

Смещение Kafka фиксируется после записи в базу, поэтому потеря сообщения
возможна, а дубль — нет: id транзакции уникален, вставка идёт через ON CONFLICT.
"""

import json
import logging
import os
import time

import psycopg2
from confluent_kafka import Consumer, KafkaError, KafkaException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scoring_writer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP_WRITER", "scoring-writer")
CONNECT_ATTEMPTS = 30
CONNECT_PAUSE_SECONDS = 2


def connect_consumer() -> Consumer:
    """Ждёт брокер и подписывается на топик результатов с ручной фиксацией."""
    config = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    last_error = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            consumer = Consumer(config)
            consumer.subscribe([SCORING_TOPIC])
            logger.info("Подписка на %s через %s", SCORING_TOPIC, KAFKA_BOOTSTRAP_SERVERS)
            return consumer
        except KafkaException as exc:
            last_error = exc
            logger.info("Kafka ещё не готова (%s/%s): %s", attempt, CONNECT_ATTEMPTS, exc)
            time.sleep(CONNECT_PAUSE_SECONDS)
    raise last_error


def connect_db():
    """Ждёт Postgres: контейнер может подниматься дольше сервиса."""
    config = {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }
    last_error = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return psycopg2.connect(**config)
        except psycopg2.OperationalError as exc:
            last_error = exc
            logger.info("Postgres ещё не готов (%s/%s): %s", attempt, CONNECT_ATTEMPTS, exc)
            time.sleep(CONNECT_PAUSE_SECONDS)
    raise last_error


def insert_score(conn, payload: dict) -> None:
    """Пишет один результат скоринга в scores.

    Повторное сообщение с тем же transaction_id обновляет строку, а не создаёт
    вторую: счётчик фродов не завышается при перечитывании топика.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO scores (transaction_id, score, fraud_flag, model_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO UPDATE
            SET score = EXCLUDED.score,
                fraud_flag = EXCLUDED.fraud_flag,
                model_version = EXCLUDED.model_version,
                created_at = now()
            """,
            (
                payload["transaction_id"],
                payload["score"],
                payload["fraud_flag"],
                payload.get("model_version"),
            ),
        )
    conn.commit()


def run() -> None:
    """Бесконечно пишет результаты скоринга в таблицу scores."""
    consumer = connect_consumer()
    conn = connect_db()
    written = 0
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
                payload = json.loads(message.value().decode("utf-8"))
                insert_score(conn, payload)
                written += 1
                if written % 100 == 0:
                    logger.info("Записано результатов: %s", written)
            except psycopg2.Error:
                conn.rollback()
                logger.exception("Запись не удалась, сообщение будет перечитано")
                continue
            except (json.JSONDecodeError, KeyError):
                logger.exception("Некорректное сообщение пропущено")
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run()
