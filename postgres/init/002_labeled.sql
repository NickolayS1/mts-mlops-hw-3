-- Витрина размеченных транзакций и указатель окна обучения.
-- Создаётся автоматически при первой инициализации контейнера Postgres.
-- Те же таблицы умеет создать data_loader, если init-скрипты не смонтированы.

CREATE TABLE IF NOT EXISTS labeled_transactions (
    transaction_time TIMESTAMP        NOT NULL,
    merch            TEXT,
    cat_id           TEXT,
    amount           DOUBLE PRECISION,
    name_1           TEXT,
    name_2           TEXT,
    gender           TEXT,
    street           TEXT,
    one_city         TEXT,
    us_state         TEXT,
    post_code        INTEGER,
    lat              DOUBLE PRECISION,
    lon              DOUBLE PRECISION,
    population_city  INTEGER,
    jobs             TEXT,
    merchant_lat     DOUBLE PRECISION,
    merchant_lon     DOUBLE PRECISION,
    target           SMALLINT
);

-- Выборка месяца для DAG переобучения и чистка старых наблюдений
CREATE INDEX IF NOT EXISTS labeled_transactions_time_idx
ON labeled_transactions (transaction_time);

-- Строка id = 1: границы окна обучения и месяц следующего батча.
-- Заполняет data_loader, сдвигает publish_model после публикации модели.
CREATE TABLE IF NOT EXISTS training_state (
    id               INT PRIMARY KEY,
    window_start     DATE NOT NULL,
    window_end       DATE NOT NULL,
    next_batch_month DATE NOT NULL,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
