"""Поток транзакций в Kafka из размеченного CSV.

Каждая отправленная транзакция получает новый id и текущее время: витрину
пополняет не он, а DAG поставки, здесь только имитация живого трафика.
"""

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from confluent_kafka import KafkaException, Producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("transaction_simulator")

CSV_PATH = Path(os.getenv("TRAIN_DATA_CSV", "/data/train.csv"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SEND_INTERVAL_SECONDS = float(os.getenv("SEND_INTERVAL_SECONDS", "0.5"))
TIME_FORMAT = "%Y-%m-%d %H:%M"
CONNECT_ATTEMPTS = 30
CONNECT_PAUSE_SECONDS = 2


def load_transactions(path: Path) -> pd.DataFrame:
    """Читает размеченный CSV один раз и держит его в памяти для выборки."""
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл транзакций: {path}")
    frame = pd.read_csv(path)
    if "transaction_time" not in frame.columns:
        raise ValueError("В CSV нет колонки transaction_time")
    logger.info("Загружено %s строк из %s", len(frame), path)
    return frame


def json_safe(value):
    """Приводит значение ячейки pandas к типу, который сериализуется в JSON."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_message(row: pd.Series) -> dict:
    """Собирает сообщение топика transactions: новый id и время «сейчас».

    Колонка target остаётся в данных: по ней потом можно сверить скор с меткой,
    признаки её не используют.
    """
    data = {column: json_safe(value) for column, value in row.items()}
    data["transaction_time"] = datetime.now().strftime(TIME_FORMAT)
    return {"transaction_id": str(uuid.uuid4()), "data": data}


def connect_producer() -> Producer:
    """Ждёт брокер и открывает продюсер.

    JSON кодируется перед отправкой вручную: сериализатор в конфигурации
    librdkafka не поддерживает, а версии обёртки принимают разные ключи.
    """
    last_error = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
        except KafkaException as exc:
            last_error = exc
            logger.info("Kafka ещё не готова (%s/%s): %s", attempt, CONNECT_ATTEMPTS, exc)
            time.sleep(CONNECT_PAUSE_SECONDS)
    raise last_error


def stream(frame: pd.DataFrame, producer: Producer) -> None:
    """Без остановки шлёт случайные строки с паузой SEND_INTERVAL_SECONDS."""
    sent = 0
    while True:
        row = frame.iloc[random.randrange(len(frame))]
        message = build_message(row)
        producer.produce(
            TRANSACTIONS_TOPIC,
            key=message["transaction_id"].encode("utf-8"),
            value=json.dumps(message).encode("utf-8"),
        )
        producer.flush()
        sent += 1
        if sent % 100 == 0:
            logger.info("Отправлено %s транзакций", sent)
        time.sleep(SEND_INTERVAL_SECONDS)


def main() -> None:
    """Загружает CSV и запускает бесконечную поставку транзакций."""
    frame = load_transactions(CSV_PATH)
    producer = connect_producer()
    logger.info(
        "Старт потока: топик %s, брокер %s, интервал %s с",
        TRANSACTIONS_TOPIC,
        KAFKA_BOOTSTRAP_SERVERS,
        SEND_INTERVAL_SECONDS,
    )
    try:
        stream(frame, producer)
    finally:
        producer.flush()


if __name__ == "__main__":
    main()
