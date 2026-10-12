-- Таблица результатов скоринга.
-- Создаётся автоматически при первой инициализации контейнера Postgres.
-- Витрину labeled_transactions и training_state создаёт data_loader.

CREATE TABLE IF NOT EXISTS scores (
    id             BIGSERIAL PRIMARY KEY,
    transaction_id TEXT             NOT NULL UNIQUE,
    score          DOUBLE PRECISION NOT NULL,
    fraud_flag     INTEGER          NOT NULL,
    model_version  TEXT,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Свежие транзакции и поиск фродов
CREATE INDEX IF NOT EXISTS scores_created_at_idx ON scores (created_at DESC);
CREATE INDEX IF NOT EXISTS scores_fraud_flag_idx ON scores (fraud_flag);
-- Проверка, что новый скор посчитан опубликованной версией
CREATE INDEX IF NOT EXISTS scores_model_version_idx ON scores (model_version);
