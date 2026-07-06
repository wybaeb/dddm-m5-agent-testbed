#!/usr/bin/env python3
"""
Инструменты агента + изоляция исполнения (Модуль 5, тема «безопасное встраивание»).

Здесь живёт «руки» агента — то, чем он может ДЕЙСТВОВАТЬ:
  • python     — выполнить код в личном каталоге студента (изолированный
                 процесс, таймаут, без интерактива); новые .png отправляются в чат;
  • list_files — показать, какие файлы есть в рабочем каталоге;
  • sql        — выполнить read-only SELECT к учебной БД (если она настроена).

Изоляция здесь учебная (подпроцесс + cwd + таймаут). В проде вместо этого
поднимают контейнер/песочницу на студента — об этом урок 5.5 «Безопасное
встраивание». Принцип один: агент действует ТОЛЬКО внутри отведённой ему
песочницы и только разрешёнными инструментами.
"""
import os
import sys
import glob
import time
import subprocess

PY_TIMEOUT = int(os.environ.get("AGENT_PY_TIMEOUT", "60"))

# --- read-only гардрейл для SQL (та же идея, что в M4.8 Safe Gen BI) ---------
SQL_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
                 "truncate", "grant", "revoke", "copy", "--", ";")


def build_tools() -> list[dict]:
    """Описания инструментов в формате OpenAI function calling."""
    return [
        {"type": "function", "function": {
            "name": "python",
            "description": (
                "Выполнить Python-код в рабочем каталоге студента. Доступны pandas, "
                "numpy, matplotlib. Возвращает stdout. Чтобы показать график — "
                "сохрани его через plt.savefig('name.png'), он уйдёт студенту картинкой."),
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "Python-код"}},
                "required": ["code"]},
        }},
        {"type": "function", "function": {
            "name": "list_files",
            "description": "Список файлов в рабочем каталоге студента (данные, результаты).",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "sql",
            "description": (
                "Выполнить read-only SELECT к учебной БД продаж (PostgreSQL). "
                "Разрешён только SELECT. Возвращает строки результата."),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "SQL-запрос (только SELECT)"}},
                "required": ["query"]},
        }},
    ]


def dispatch_tool(name: str, args: dict, workspace: str):
    """Выполнить инструмент. Возвращает (текст_наблюдения, [новые png])."""
    if name == "python":
        return _run_python(args.get("code", ""), workspace)
    if name == "list_files":
        return _list_files(workspace), []
    if name == "sql":
        return _run_sql(args.get("query", "")), []
    return f"Неизвестный инструмент: {name}", []


def _snapshot_pngs(workspace: str) -> dict:
    return {p: os.path.getmtime(p) for p in glob.glob(os.path.join(workspace, "*.png"))}


def _run_python(code: str, workspace: str):
    """Запустить код изолированным процессом в каталоге студента, поймать новые png."""
    os.makedirs(workspace, exist_ok=True)
    before = _snapshot_pngs(workspace)

    script = os.path.join(workspace, "_agent_cell.py")
    # matplotlib без дисплея + запрет случайных окон
    preamble = "import matplotlib\nmatplotlib.use('Agg')\n"
    with open(script, "w") as f:
        f.write(preamble + code)

    try:
        proc = subprocess.run(
            [sys.executable, "_agent_cell.py"],
            cwd=workspace, capture_output=True, text=True, timeout=PY_TIMEOUT,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        out = (proc.stdout or "")
        if proc.returncode != 0:
            out += "\n[stderr]\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return f"⏱ Код выполнялся дольше {PY_TIMEOUT}с и был остановлен. Упрости задачу.", []
    finally:
        if os.path.exists(script):
            os.remove(script)

    after = _snapshot_pngs(workspace)
    new_images = [p for p, m in after.items() if before.get(p) != m]

    out = out.strip() or "(код выполнен, вывода нет — добавь print() или сохрани график)"
    if new_images:
        out += f"\n[создано графиков: {len(new_images)}]"
    return out[:4000], new_images


def _list_files(workspace: str) -> str:
    files = sorted(os.listdir(workspace)) if os.path.isdir(workspace) else []
    files = [f for f in files if not f.startswith("_agent_cell")]
    if not files:
        return "Рабочий каталог пуст. Данные можно загрузить через задание или скриптом."
    lines = []
    for f in files:
        path = os.path.join(workspace, f)
        size = os.path.getsize(path)
        lines.append(f"  {f}  ({size//1024} КБ)" if size > 1024 else f"  {f}  ({size} Б)")
    return "Файлы в рабочем каталоге:\n" + "\n".join(lines)


def _run_sql(query: str) -> str:
    """Read-only SELECT к учебному Postgres. Если БД не настроена — честно сообщаем."""
    q = query.strip().rstrip(";")
    low = q.lower()
    if not low.startswith(("select", "with")):
        return "⛔ Разрешён только SELECT (read-only). Запрос отклонён гардрейлом."
    for bad in SQL_FORBIDDEN:
        if bad in low.split():
            return f"⛔ Запрещённое ключевое слово '{bad}'. Запрос отклонён гардрейлом."
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return ("БД не подключена (нет DATABASE_URL). Для аналитики используй "
                "инструмент python и CSV-файлы из рабочего каталога.")
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(dsn)
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM ({q}) _sub LIMIT 100")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        conn.close()
        head = " | ".join(cols)
        body = "\n".join(" | ".join(str(c) for c in r) for r in rows[:50])
        return f"{head}\n{body}\n[строк: {len(rows)}]"
    except Exception as e:  # noqa: BLE001
        return f"Ошибка БД: {e}"
