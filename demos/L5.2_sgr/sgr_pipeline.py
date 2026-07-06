#!/usr/bin/env python3
"""
Урок 5.2 — Schema Guided Reasoning (SGR): когда агенту нужен жёсткий сценарий.

ReAct (урок 5.1) свободен: модель сама решает, сколько шагов сделать и какие.
Это гибко, но непредсказуемо — для регламентированных бизнес-процессов (отчёт по
шаблону, проверка по чек-листу) важнее ПОВТОРЯЕМОСТЬ. SGR задаёт жёсткую СХЕМУ
рассуждения: модель обязана заполнить заранее определённые поля, а вычисление
выполняется детерминированно нашим кодом, а не «как придумает агент».

Здесь схема: restate → нужные_столбцы → одно pandas-выражение → единица измерения.
Мы исполняем выражение сами и просим модель только истолковать результат.
Один проход, ноль свободных циклов — предсказуемо и аудируемо.

Запуск:
    export AI_PROVIDER_TOKEN=...
    python3 sgr_pipeline.py "Средний чек по каналам привлечения"
"""
import os
import sys
import json

import pandas as pd
from openai import OpenAI

BASE_URL = os.environ.get("AI_PROVIDER_URL", "https://api.agentplatform.ru/v1")
API_KEY = os.environ.get("AI_PROVIDER_TOKEN", "")
MODEL = os.environ.get("AI_PROVIDER_MODEL", "openai/gpt-4o")
DATA_PATH = os.environ.get("SGR_DATA", os.path.join(
    os.path.dirname(__file__), "..", "shared", "data", "sales_data.csv"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Жёсткая схема рассуждения — модель ОБЯЗАНА вернуть ровно эти поля
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "restate": {"type": "string", "description": "переформулировка задачи"},
        "columns": {"type": "array", "items": {"type": "string"},
                    "description": "столбцы, которые нужны"},
        "pandas_expr": {"type": "string",
            "description": "ОДНО выражение над DataFrame df, возвращающее результат "
                           "(Series/DataFrame/скаляр). Без присваиваний, без print."},
        "unit": {"type": "string", "description": "единица измерения результата"},
    },
    "required": ["restate", "columns", "pandas_expr", "unit"],
    "additionalProperties": False,
}

SCHEMA_HINT = ("Столбцы df: order_id, date, customer_id, city, channel, "
               "product_category, product, quantity, price, revenue.")


def make_plan(question: str) -> dict:
    sys_p = ("Ты заполняешь жёсткую схему анализа (Schema Guided Reasoning). "
             "Верни СТРОГО JSON ровно с этими ключами верхнего уровня и ничем больше: "
             '"restate" (строка), "columns" (массив строк), "pandas_expr" (строка), '
             '"unit" (строка). pandas_expr — одно безопасное выражение над df '
             "(groupby/agg/sort и т.п.), без присваиваний и импортов. " + SCHEMA_HINT)
    msg = client.chat.completions.create(
        model=MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_p},
                  {"role": "user", "content": question}],
    ).choices[0].message
    return json.loads(msg.content)


def execute(expr: str, df: pd.DataFrame):
    """Детерминированное исполнение — НЕ модель, а наш код. Только чтение df."""
    if any(t in expr for t in ("=", "import", "__", "open(", "exec", "eval")):
        raise ValueError("Выражение нарушает ограничения SGR (запрещены присваивания/импорты).")
    return eval(expr, {"df": df, "pd": pd})  # noqa: S307 — выражение ограничено схемой


def interpret(question: str, plan: dict, result) -> str:
    msg = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": "Истолкуй результат расчёта деловым языком, 2-3 предложения."},
                  {"role": "user", "content": f"Вопрос: {question}\nРезультат ({plan['unit']}):\n{result}"}],
    ).choices[0].message
    return msg.content.strip()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Средний чек по каналам привлечения"
    if not API_KEY:
        print("⚠️  Нет AI_PROVIDER_TOKEN — ключ LLM-роутера выдаёт LMS."); sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print(f"❓ {question}\n{'─'*60}")
    plan = make_plan(question)
    # SGR допускает только заданные поля; нормализуем на случай вольностей модели
    plan = {"restate": plan.get("restate", question),
            "columns": plan.get("columns", []),
            "pandas_expr": plan.get("pandas_expr") or plan.get("expr", ""),
            "unit": plan.get("unit", "")}
    if not plan["pandas_expr"]:
        print("⚠️  Модель не вернула pandas_expr по схеме — SGR отклоняет ответ."); sys.exit(1)
    print("📋 СХЕМА РАССУЖДЕНИЯ (заполнена моделью):")
    for k in ("restate", "columns", "pandas_expr", "unit"):
        print(f"   {k}: {plan[k]}")
    print("─" * 60)
    result = execute(plan["pandas_expr"], df)
    print("🧮 РЕЗУЛЬТАТ (исполнено детерминированно нашим кодом):")
    print(result)
    print("─" * 60)
    print("🟣 ВЫВОД:", interpret(question, plan, result))
