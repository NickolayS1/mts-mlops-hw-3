"""DAG поставки очередного месяца в размеченную витрину.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.supply.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.supply.append_labeled_batch import append_labeled_batch
from pipeline.supply.prune_old_months import prune_old_months
from pipeline.supply.sample_next_month import sample_next_month


@dag(
    dag_id="supply_labeled_batch",
    description="Раз в 5 минут дописывает в витрину следующий месяц наблюдений",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data"],
    doc_md="""
### Поставка размеченного месяца

1. Целевой месяц берётся из `training_state.next_batch_month` — ровно тот,
   который следующим возьмёт `retrain_catboost`.
2. Из витрины случайно выбирается до `SAMPLE_SIZE` строк, их время переносится
   в целевой месяц.
3. Батч дописывается в `labeled_transactions`.
4. Наблюдения старше `HISTORY_YEARS` лет от самого нового месяца удаляются.

Витрина опережает окно обучения на один месяц, поэтому чистка начинает
удалять старые наблюдения. Если месяц уже в витрине, дозапись помечается
skipped, поэтому частый запуск не дублирует наблюдения.
    """,
)
def supply_labeled_batch():
    """Связывает подготовку батча, дозапись в витрину и чистку старых месяцев."""
    prepared = task(sample_next_month, task_id="sample_next_month")()
    appended = task(append_labeled_batch, task_id="append_labeled_batch")(prepared)
    # Чистка зависит от подготовки, а не от дозаписи: когда месяц уже в витрине,
    # дозапись пропускается, но удалять старые наблюдения всё равно нужно.
    task(prune_old_months, task_id="prune_old_months")(prepared)
    appended


supply_labeled_batch()
