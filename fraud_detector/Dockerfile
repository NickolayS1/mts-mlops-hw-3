FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Схема Ф2 общая с Airflow: пакет монтируется в /app/pipeline, см. docker-compose.
ENV PYTHONUNBUFFERED=1

CMD ["python", "./app/app.py"]
