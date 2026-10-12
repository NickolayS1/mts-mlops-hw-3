"""DAG скользящего переобучения CatBoost.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.retrain.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain.evaluate_against_current import evaluate_against_current
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.publish_model import publish_model
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split


@dag(
    dag_id="retrain_catboost",
    description="Раз в 5 минут сдвигает окно обучения и переобучает CatBoost",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "catboost"],
    doc_md="""
### Переобучение CatBoost

Каждый запуск имитирует новый месяц в проде:

1. Из витрины `labeled_transactions` берётся месяц из `training_state.next_batch_month`.
2. Месяц добавляется в окно, самый старый месяц окна выбрасывается.
3. Train — ранние месяцы окна, test — самый свежий.
4. Признаки считает общая схема Ф2 (`pipeline/features.py`), та же, что в скоринге.
5. Версия публикуется в общий каталог моделей только если ROC-AUC не хуже
   предыдущей версии; иначе задача проверки помечается skipped и активной
   остаётся прежняя модель.

Указатель месяца сдвигается последним шагом, поэтому упавший прогон не
пропускает месяц. Если очередного месяца в витрине ещё нет, задачи помечаются
skipped; новые месяцы дописывает DAG `supply_labeled_batch`.
    """,
)
def retrain_catboost():
    """Связывает извлечение месяца, сдвиг окна, split, обучение, проверку и публикацию."""
    extracted = task(extract_month_batch, task_id="extract_month_batch")()
    slid = task(slide_training_window, task_id="slide_training_window")(extracted)
    split = task(time_train_test_split, task_id="time_train_test_split")(slid)
    trained = task(fit_catboost, task_id="fit_catboost")(split)
    accepted = task(evaluate_against_current, task_id="evaluate_against_current")(trained)
    task(publish_model, task_id="publish_model")(accepted)


retrain_catboost()
