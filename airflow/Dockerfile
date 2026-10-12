FROM apache/airflow:2.10.5-python3.11

USER airflow
# Версии зафиксированы: их обновление ломает Airflow 2.10, а catboost 1.2.8
# должен совпадать с версией в образе сервиса скоринга, иначе модель не загрузится.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "pandas==2.1.4" \
    "scipy==1.14.1" \
    "joblib==1.4.2" \
    "scikit-learn==1.5.2" \
    "catboost==1.2.8" \
    "cloudpickle==2.2.1" \
    "psycopg2-binary==2.9.10" \
    "python-dateutil==2.9.0.post0" \
    "pyarrow==17.0.0" \
    && python -c "import catboost, sklearn, pandas, pyarrow, psycopg2; print(catboost.__version__, pandas.__version__, pyarrow.__version__)"

USER root
COPY entrypoint.sh /custom-entrypoint.sh
RUN chmod +x /custom-entrypoint.sh

ENTRYPOINT ["/custom-entrypoint.sh"]
