"""Проверка, что все три DAG собираются и задачи связаны в ожидаемый граф.

Запуск из venv с Airflow. Нужен AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: DagBag
создаёт метабазу Airflow, поэтому используется временный SQLite-файл.
Запуск: <venv>/Scripts/python.exe scripts/check_dags.py
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

os.environ.setdefault(
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "sqlite:///" + str(Path(tempfile.gettempdir()) / "dsh_airflow_check.db"),
)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER", str(ROOT / "dags"))
os.environ.setdefault("AIRFLOW_HOME", str(Path(tempfile.gettempdir()) / "dsh_airflow_home"))

EXPECTED = {
    "retrain_catboost": [
        "extract_month_batch",
        "slide_training_window",
        "time_train_test_split",
        "fit_catboost",
        "evaluate_against_current",
        "publish_model",
    ],
    "supply_labeled_batch": [
        "sample_next_month",
        "append_labeled_batch",
        "prune_old_months",
    ],
    "check_dataset_quality": ["check_dataset_quality"],
}


def main() -> None:
    from airflow.models.dagbag import DagBag

    bag = DagBag(dag_folder=str(ROOT / "dags"), include_examples=False)
    if bag.import_errors:
        for path, error in bag.import_errors.items():
            print(f"ОШИБКА импорта {path}:\n{error}")
        raise SystemExit("DAG не импортируются")

    print(f"загружено DAG: {sorted(bag.dag_ids)}")
    for dag_id, expected_tasks in EXPECTED.items():
        # bag.dags вместо bag.get_dag: второй обращается к метабазе Airflow.
        dag = bag.dags.get(dag_id)
        if dag is None:
            raise SystemExit(f"нет DAG {dag_id}")
        tasks = sorted(dag.task_ids)
        print(f"\n{dag_id}: {tasks}")
        print(f"  расписание: {dag.schedule_interval}, catchup={dag.catchup}, "
              f"max_active_runs={dag.max_active_runs}")
        edges = sorted(
            f"{task_id} -> {downstream}"
            for task_id, task in dag.task_dict.items()
            for downstream in task.downstream_task_ids
        )
        print(f"  зависимости: {edges}")
        assert tasks == sorted(expected_tasks), f"{dag_id}: неожиданный набор задач"
        assert dag.catchup is False, f"{dag_id}: catchup должен быть выключен"
        assert dag.schedule_interval is not None, f"{dag_id}: нет расписания"
        for task_id in expected_tasks:
            if task_id in ("extract_month_batch", "sample_next_month", "check_dataset_quality"):
                continue
            upstream = dag.get_task(task_id).upstream_task_ids
            assert upstream, f"{dag_id}.{task_id}: задача ни от чего не зависит"

    print("\nвсе DAG собраны, задачи связаны")


if __name__ == "__main__":
    main()
