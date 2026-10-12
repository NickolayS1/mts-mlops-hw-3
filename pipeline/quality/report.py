"""HTML-отчет по правилам Data Quality."""

import html
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.quality import rules
from pipeline.settings import CURRENT_DATASET, DQ_REPORTS_DIR

logger = logging.getLogger(__name__)

PASS_COLOR = "#1a7f37"
FAIL_COLOR = "#b42318"


def reference_part(frame: pd.DataFrame) -> pd.DataFrame:
    """Возвращает более ранние месяцы окна как референс известных значений.

    Последний месяц окна проверяется по предыдущим: так новые мерчанты и
    категории видно, а проверка не становится тавтологией на самих себе.
    """
    periods = pd.to_datetime(frame["transaction_time"]).dt.to_period("M")
    latest = periods.max()
    earlier = frame.loc[periods < latest]
    return earlier if not earlier.empty else frame


def render(checks: list[dict], dataset: Path, rows: int) -> str:
    """Собирает HTML-страницу отчета по результатам правил."""
    failed = [check for check in checks if not check["passed"]]
    verdict = "все проверки пройдены" if not failed else f"провалено правил: {len(failed)}"
    color = PASS_COLOR if not failed else FAIL_COLOR
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows_html = "\n".join(
        "<tr>"
        f"<td class='rule'>{html.escape(check['rule'])}</td>"
        f"<td style='color:{PASS_COLOR if check['passed'] else FAIL_COLOR};font-weight:600'>"
        f"{'OK' if check['passed'] else 'FAIL'}</td>"
        f"<td>{html.escape(str(check['observed']))}</td>"
        f"<td>{html.escape(str(check['threshold']))}</td>"
        f"<td>{html.escape(check['message'])}</td>"
        "</tr>"
        for check in checks
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Data Quality report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 32px; color: #1f2328; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #57606a; font-size: 13px; margin-bottom: 20px; }}
  .verdict {{ font-size: 16px; font-weight: 600; color: {color}; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  td.rule {{ font-family: Consolas, monospace; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Data Quality проверки набора обучения</h1>
<div class="meta">Набор: {html.escape(str(dataset))} · строк: {rows} · отчёт собран: {generated}</div>
<div class="verdict">{verdict}</div>
<table>
<thead>
<tr><th>Правило</th><th>Итог</th><th>Наблюдение</th><th>Требование</th><th>Комментарий</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""


def write_report(checks: list[dict], dataset: Path, rows: int) -> Path:
    """Пишет отчет в каталог dq_reports и возвращает путь.

    Каталог создаётся при необходимости, имя файла содержит время: отчеты
    разных прогонов не затирают друг друга. Повторный запуск создаёт новый файл
    и ничего не удаляет.
    """
    DQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = DQ_REPORTS_DIR / f"dq_report_{stamp}.html"
    path.write_text(render(checks, dataset, rows), encoding="utf-8")
    logger.info("DQ-отчет записан: %s", path)
    return path


def check_dataset_quality() -> dict:
    """Проверяет текущее окно обучения и публикует HTML-отчет.

    Читает current.parquet, референс берёт из более ранних месяцев того же окна.
    Пишет HTML в DQ_REPORTS_DIR; существующие отчеты не трогает. Возвращает
    путь к отчету, число проверок и список имён проваленных правил. Падение
    задачи не закладывается: отчет и есть результат проверки.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет окна обучения: {CURRENT_DATASET}")
    frame = pd.read_parquet(CURRENT_DATASET)
    reference = reference_part(frame)
    moments = pd.to_datetime(frame["transaction_time"])
    checks = rules.run_all(frame, reference, moments.min(), moments.max())
    path = write_report(checks, CURRENT_DATASET, len(frame))
    failed = [check["rule"] for check in checks if not check["passed"]]
    logger.info("DQ-проверки: всего %s, провалено %s", len(checks), len(failed))
    return {
        "report_path": str(path),
        "checks": len(checks),
        "failed": failed,
    }
