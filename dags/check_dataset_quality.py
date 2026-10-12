"""DAG проверок датасета на требования Data Quality.

Файл только связывает задачи. Правила лежат в pipeline.quality.rules,
сборка HTML-отчета — в pipeline.quality.report.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.quality.report import check_dataset_quality


@dag(
    dag_id="check_dataset_quality",
    description="Раз в 5 минут проверяет окно обучения и публикует HTML-отчёт",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data-quality"],
    doc_md="""
### Проверки датасета

Правила описаны в README и реализованы в `pipeline/quality/rules.py`:
обязательные колонки, пропуски, период наблюдений, значения метки, доля фрода,
положительная сумма, дубли, известные категориальные значения, новые мерчанты.

Результат каждого прогона — HTML-отчёт в каталоге `dq_reports/` рядом с
каталогом моделей, файлы не перезаписываются. Провал правил не валит задачу:
отчёт со статусом FAIL и есть результат проверки.
    """,
)
def check_dataset_quality_dag():
    """Прогоняет проверки датасета и публикует отчёт."""
    task(check_dataset_quality, task_id="check_dataset_quality")()


check_dataset_quality_dag()
