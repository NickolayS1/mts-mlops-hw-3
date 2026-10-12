"""Настройки сервиса скоринга и общий словарь схемы Ф2."""

import os
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))


# Каталог в образе с начальной парой модель+словарь. Копируется в MODELS_DIR,
# только пока DAG не опубликовал свою версию.
BOOTSTRAP_DIR = Path(os.getenv("BOOTSTRAP_DIR", "/app/models"))
LATEST_POINTER = "latest.txt"

# Ожидание начальной пары: data_loader обучает её дольше, чем стартует сервис.
BOOTSTRAP_ATTEMPTS = 60
BOOTSTRAP_PAUSE_SECONDS = 5

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")
CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "ml-scorer")

# Порог перевода вероятности в fraud_flag, как в семинаре.
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
