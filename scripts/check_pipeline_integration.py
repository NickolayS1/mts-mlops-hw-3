"""Интеграционная проверка витрины, поставки и переобучения на живой Postgres.

Прогоняет тот же код, что вызывают DAG: data_loader/steps.py, pipeline.supply.*
и pipeline.retrain.*. Ожидает Postgres на POSTGRES_HOST/POSTGRES_PORT и каталог
для артефактов в TRAINING_DATA_DIR/MODELS_DIR.

Запускать из каталога вне репозитория (cwd не должен быть корнем проекта) и
с ключом -P: иначе каталог airflow/ в корне перекрывает установленный Airflow
как namespace-пакет, и импорт pipeline.retrain падает на airflow.exceptions.
Запуск: <venv>/Scripts/python.exe -P <путь>/scripts/check_pipeline_integration.py
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_loader"))
# Корень проекта ставим в конец sys.path: так пакет pipeline находится, а
# пустой каталог airflow/ из корня не перекрывает настоящий Airflow из venv.
sys.path.append(str(ROOT))

ARTIFACTS = Path(tempfile.mkdtemp(prefix="hw3_artifacts_"))
TRAINING_DIR = ARTIFACTS / "training_data"
MODELS_DIR = ARTIFACTS / "models"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("POSTGRES_DB", "fraud_db")
os.environ.setdefault("POSTGRES_USER", "fraud_user")
os.environ.setdefault("POSTGRES_PASSWORD", "fraud_pass")
os.environ["TRAINING_DATA_DIR"] = str(TRAINING_DIR)
os.environ["MODELS_DIR"] = str(MODELS_DIR)

import pipeline.db as db  # noqa: E402
import pipeline.settings as settings  # noqa: E402
import pipeline.train as train  # noqa: E402
from airflow.exceptions import AirflowSkipException  # noqa: E402
from pipeline.retrain.evaluate_against_current import (  # noqa: E402
    decide,
    evaluate_against_current,
)
from pipeline.retrain import extract_month_batch, fit_catboost, publish_model  # noqa: E402
from pipeline.retrain import slide_training_window, time_train_test_split  # noqa: E402
from pipeline.supply import append_labeled_batch, prune_old_months, sample_next_month  # noqa: E402

CSV_SUBSET = ARTIFACTS / "window.csv"


def dataset_path() -> Path:
    """Путь к CSV: аргумент командной строки, затем TRAIN_DATA_CSV."""
    parser = argparse.ArgumentParser(description="Интеграционная проверка пайплайна")
    parser.add_argument("csv", nargs="?", default=os.getenv("TRAIN_DATA_CSV"))
    args = parser.parse_args()
    if not args.csv:
        raise SystemExit("Укажите путь к train.csv аргументом или в TRAIN_DATA_CSV")
    return Path(args.csv)


def patch_paths() -> None:
    """Направляет все модули в каталог артефактов теста."""
    for module in (
        settings,
        db,
        train,
        extract_month_batch,
        fit_catboost,
        publish_model,
        sample_next_month,
        append_labeled_batch,
        prune_old_months,
        slide_training_window,
        time_train_test_split,
    ):
        for name, value in (
            ("TRAINING_DIR", TRAINING_DIR),
            ("MODELS_DIR", MODELS_DIR),
            ("CURRENT_DATASET", TRAINING_DIR / "current.parquet"),
            ("PENDING_WINDOW", TRAINING_DIR / "pending_window.parquet"),
            ("BATCHES_DIR", TRAINING_DIR / "batches"),
            ("SPLITS_DIR", TRAINING_DIR / "splits"),
            ("SUPPLY_BATCH", TRAINING_DIR / "supply_batch.parquet"),
        ):
            if hasattr(module, name):
                setattr(module, name, value)
    fit_catboost.CURRENT_DATASET = TRAINING_DIR / "current.parquet"


def query(sql: str, params=None) -> list:
    """Выполняет запрос и возвращает строки: нужен для сверки состояния витрины."""
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def state() -> dict:
    """Текущее состояние окна обучения."""
    rows = query("SELECT window_start, window_end, next_batch_month FROM training_state WHERE id = 1")
    window_start, window_end, next_batch = rows[0]
    return {"start": window_start, "end": window_end, "next": next_batch}


def counts_by_month() -> dict:
    """Число наблюдений по месяцам витрины."""
    rows = query(
        """
        SELECT to_char(date_trunc('month', transaction_time), 'YYYY-MM'), count(*)
        FROM labeled_transactions
        GROUP BY 1 ORDER BY 1
        """
    )
    return {month: count for month, count in rows}


def main() -> None:
    patch_paths()
    db.get_conn()  # ранняя диагностика: без Postgres тест бессмысленен

    # Тест должен быть воспроизводимым: на повторном запуске витрина и указатель
    # месяца возвращаются в исходное состояние.
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE labeled_transactions")
            cur.execute("TRUNCATE training_state")
        conn.commit()
    finally:
        conn.close()

    # Витрина наполняется тремя месяцами: этого хватает на поставку месяца из
    # указателя, переобучение и проверку чистки.
    csv = pd.read_csv(dataset_path(), parse_dates=["transaction_time"])
    subset = csv.loc[csv["transaction_time"] < pd.Timestamp("2019-04-01")]
    subset.to_csv(CSV_SUBSET, index=False)
    print(f"подготовлен CSV для загрузки: {len(subset)} строк, "
          f"месяцы {sorted(subset['transaction_time'].dt.to_period('M').unique().astype(str))}")

    import steps
    import train_starter

    steps.CSV_PATH = CSV_SUBSET
    train_starter.CURRENT_DATASET = TRAINING_DIR / "current.parquet"

    conn = steps.connect()
    try:
        steps.ensure_schema(conn)
        loaded = steps.load_csv(conn)
        steps.ensure_training_state(conn)
        steps.materialize_window(conn)
    finally:
        conn.close()

    print(f"\nшаг 1: загружено {loaded} строк")
    print(f"  состояние окна: {state()}")
    print(f"  строк по месяцам: {counts_by_month()}")
    assert state()["next"].isoformat() == "2019-04-01", "первый батч DAG должен быть 2019-04"

    print(f"\nшаг 2: стартовая модель")
    version = train_starter.train_starter()
    print(f"  версия {version}, файлы {sorted(path.name for path in MODELS_DIR.iterdir())}")
    assert version == "initial"
    assert (MODELS_DIR / "model_initial.cbm").exists()
    assert (MODELS_DIR / "preproc_initial.json").exists()
    assert (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip() == "initial"

    print("\nшаг 3: поставка месяца 2019-04")
    before = counts_by_month()
    prepared = sample_next_month.sample_next_month()
    print(f"  батч {prepared['batch_month']}: {prepared['rows']} строк, "
          f"в витрине уже было {prepared['already_present']}")
    appended = append_labeled_batch.append_labeled_batch(prepared)
    pruned = prune_old_months.prune_old_months(appended)
    after = counts_by_month()
    print(f"  строк месяца в витрине после поставки: {after.get('2019-04', 0)}")
    print(f"  чистка: удалено {pruned['deleted_rows']} строк, граница {pruned['cutoff']}")
    assert prepared["batch_month"] == "2019-04-01"
    assert after.get("2019-04", 0) > 0, "месяц поставки не появился в витрине"
    assert sum(after.values()) == sum(before.values()) + prepared["rows"], "дозапись потеряла строки"

    print("\nшаг 4: повторный запуск поставки того же месяца")
    again = sample_next_month.sample_next_month()
    try:
        append_labeled_batch.append_labeled_batch(again)
        raise AssertionError("повторная дозапись того же месяца должна быть пропущена")
    except AirflowSkipException:
        print("  дозапись пропущена, как и ожидалось")
    assert sum(counts_by_month().values()) == sum(after.values()), "повтор удвоил наблюдения"

    print("\nшаг 5: переобучение на месяце 2019-04")
    extracted = extract_month_batch.extract_month_batch()
    print(f"  батч из витрины: {Path(extracted['batch_path']).name}, месяц {extracted['batch_month']}")
    assert extracted["batch_month"] == "2019-04-01"

    slid = slide_training_window.slide_training_window(extracted)
    print(f"  окно: {Path(slid['dataset_path']).name}, удалён месяц {slid['dropped_month']}")
    assert slid["dropped_month"] == "2019-01", "должен уйти самый старый месяц окна"
    split = time_train_test_split.time_train_test_split(slid)
    print(f"  train месяцы {split['train_months']}, test {split['test_month']}")
    assert split["test_month"] == "2019-04"

    trained = fit_catboost.fit_catboost(split)
    print(f"  кандидат {trained['version']}: ROC-AUC={trained['metrics']['roc_auc']:.4f}, "
          f"F1={trained['metrics']['f1']:.4f}")
    print(f"  активная версия до публикации: {trained['current_version']}")
    assert (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip() == "initial", \
        "обучение не должно менять указатель до проверки"
    assert (MODELS_DIR / f".staging_model_{trained['version']}.cbm").exists(), "нет черновика модели"

    accepted = evaluate_against_current(trained)
    print(f"  проверка улучшения: публиковать={accepted['promote']} ({accepted['reason']})")
    published = publish_model.publish_model(accepted)
    active = (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
    print(f"  опубликовано: {published['published']}, активная версия {active}")
    print(f"  состояние окна: {state()}")
    print(f"  файлы: {sorted(path.name for path in MODELS_DIR.iterdir())}")
    assert active == trained["version"], "указатель не переключился на новую версию"
    assert state()["next"].isoformat() == "2019-05-01", "указатель месяца не сдвинулся"
    assert state()["start"].isoformat() == "2019-02-01", "окно не сдвинулось на месяц"
    assert (TRAINING_DIR / "current.parquet").exists()

    print("\nшаг 6: уровень DAG оставил только нужные версии")
    stored = sorted(path.stem[len("model_") :] for path in MODELS_DIR.glob("model_*.cbm"))
    print(f"  версии моделей на диске: {stored}, удалено: {published['removed_versions']}")
    assert len(stored) <= settings.KEEP_VERSIONS, "устаревшие версии не удалены"

    print("\nшаг 7: чистка удаляет наблюдения старше глубины истории")
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO labeled_transactions (transaction_time, merch, amount, target) "
                "VALUES ('2017-01-15 12:00', 'fraud_Старый', 100.0, 0)"
            )
        conn.commit()
    finally:
        conn.close()
    old_rows = query("SELECT count(*) FROM labeled_transactions WHERE transaction_time < '2018-01-01'")[0][0]
    pruned_again = prune_old_months.prune_old_months({"batch_month": "2019-05-01"})
    left = query("SELECT count(*) FROM labeled_transactions WHERE transaction_time < '2018-01-01'")[0][0]
    print(f"  старых строк до чистки: {old_rows}, удалено: {pruned_again['deleted_rows']}, осталось: {left}")
    print(f"  граница хранения {pruned_again['cutoff']} от месяца {pruned_again['newest_month']}")
    assert old_rows == 1 and left == 0, "чистка не удалила старое наблюдение"

    print("\nшаг 8: скоринг опубликованной парой")
    from catboost import CatBoostClassifier
    from pipeline.features import build_features, feature_matrix, load_preproc

    active_version = (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / f"model_{active_version}.cbm"))
    preproc = load_preproc(MODELS_DIR / f"preproc_{active_version}.json")
    assert preproc["version"] == active_version, "словарь не совпал с версией модели"
    sample = pd.read_parquet(split["test_raw_path"]).head(200)
    probabilities = model.predict_proba(feature_matrix(build_features(sample, preproc)))[:, 1]
    flags = (probabilities > 0.5).astype(int)
    print(f"  версия {active_version}: {len(probabilities)} транзакций, "
          f"фродов {int(flags.sum())}, максимум скора {probabilities.max():.4f}")
    assert len(probabilities) == 200 and probabilities.min() >= 0 and probabilities.max() <= 1

    print("\nшаг 9: решение о публикации на синтетических метриках")
    version = trained["version"]
    before = (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
    cases = (
        ("первая версия", {"roc_auc": 0.90}, None, True),
        ("заметно лучше", {"roc_auc": 0.99}, {"roc_auc": 0.95}, True),
        ("в пределах допуска", {"roc_auc": 0.948}, {"roc_auc": 0.95}, True),
        ("заметно хуже", {"roc_auc": 0.80}, {"roc_auc": 0.95}, False),
        ("нет метрики у активной", {"roc_auc": 0.90}, {}, True),
    )
    for label, metrics, previous, expected in cases:
        promote, reason = decide(metrics, previous)
        print(f"  {label:22} -> {'публиковать' if promote else 'не публиковать'}: {reason}")
        assert promote is expected, f"{label}: неверное решение проверки улучшения"

    rejected = {**accepted, "promote": False, "metrics": {"version": version, "roc_auc": 0.8}}
    try:
        publish_model.publish_model(rejected)
        raise AssertionError("публикация отклонённой версии должна падать")
    except RuntimeError as exc:
        print(f"  публикация отклонённой версии не выполнена: {type(exc).__name__}")
    assert (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip() == before, \
        "указатель сдвинулся при отклонённой публикации"

    print("\nшаг 10: проверки Data Quality на текущем окне")
    from pipeline.quality import report as quality

    quality.CURRENT_DATASET = TRAINING_DIR / "current.parquet"
    quality.DQ_REPORTS_DIR = ARTIFACTS / "dq_reports"
    outcome = quality.check_dataset_quality()
    report_path = Path(outcome["report_path"])
    text = report_path.read_text(encoding="utf-8")
    print(f"  отчёт: {report_path.name}, правил {outcome['checks']}, провалено {len(outcome['failed'])}")
    print(f"  проваленные: {outcome['failed'] or 'нет'}")
    assert report_path.exists() and "<table>" in text, "HTML-отчёт не собран"
    assert outcome["checks"] >= 5, "правил DQ должно быть не меньше пяти"
    assert not outcome["failed"], f"правила DQ не должны падать на реальных данных: {outcome['failed']}"

    print(f"\nартефакты теста: {ARTIFACTS}")
    print("интеграция витрины, поставки и переобучения проверена")


if __name__ == "__main__":
    main()
