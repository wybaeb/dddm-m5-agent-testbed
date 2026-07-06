#!/usr/bin/env python3
"""
Урок 5.4 — Claw-like агент: автономное планирование → исполнение → проверка → итерация.

Отличие от чистого ReAct (урок 5.1): ReAct реагирует шаг-за-шагом «на ходу».
Claw-like агент сначала строит ПЛАН из подзадач, затем выполняет их, а в конце
сам ПРОВЕРЯЕТ результат и при необходимости повторяет цикл. Это уже автономное
поведение: вы ставите цель, агент сам ведёт исследование.

Цикл здесь:
    PLAN     → LLM раскладывает цель на 2–4 шага
    EXECUTE  → каждый шаг выполняется ReAct-под-агентом (python над данными)
    VERIFY   → LLM проверяет: достигнута ли цель, всё ли посчитано на данных
    (если нет — REPLAN с учётом замечаний, максимум N итераций)

Запуск:
    export AI_PROVIDER_TOKEN=...
    python3 claw_agent.py "Разберись, почему в одном из месяцев просела выручка"
"""
import io
import os
import sys
import json
import contextlib
import traceback

from openai import OpenAI

BASE_URL = os.environ.get("AI_PROVIDER_URL", "https://api.agentplatform.ru/v1")
API_KEY = os.environ.get("AI_PROVIDER_TOKEN", "")
MODEL = os.environ.get("AI_PROVIDER_MODEL", "openai/gpt-4o")
DATA_PATH = os.environ.get("CLAW_DATA", os.path.join(
    os.path.dirname(__file__), "..", "shared", "data", "sales_data.csv"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
C = {"plan": "\033[1;36m", "exec": "\033[33m", "verify": "\033[1;35m",
     "dim": "\033[2m", "reset": "\033[0m"}

# Схема учебного датасета — даём агенту заранее, чтобы он не угадывал имена колонок
SCHEMA = ("CSV-файл по пути DATA_PATH, столбцы: order_id, date (YYYY-MM-DD), "
          "customer_id, city, channel, product_category, product, quantity, "
          "price, revenue. Загрузка: df = pd.read_csv(DATA_PATH); "
          "df['date'] = pd.to_datetime(df['date']).")


def _chat(messages, **kw):
    return client.chat.completions.create(model=MODEL, messages=messages,
                                          temperature=0, **kw).choices[0].message


def run_python(code: str) -> str:
    import pandas as pd
    ns = {"pd": pd, "DATA_PATH": DATA_PATH}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        return buf.getvalue() + "\n" + traceback.format_exc()
    return buf.getvalue().strip() or "(нет вывода)"


PY_TOOL = [{"type": "function", "function": {
    "name": "python",
    "description": "Выполнить Python-код, вернуть print(). Доступны pd и DATA_PATH.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}}, "required": ["code"]}}}]


def plan(goal: str, feedback: str = "") -> list[str]:
    """Шаг PLAN: разложить цель на конкретные подзадачи."""
    sys_p = ("Ты планировщик автономного аналитического агента. Разложи цель на "
             "2–4 КОНКРЕТНЫХ проверяемых шага анализа данных. Верни JSON "
             '{"steps": ["...", "..."]}. Без воды, каждый шаг — измеримое действие.')
    user = f"Цель: {goal}\nДанные: {SCHEMA}"
    if feedback:
        user += f"\n\nПредыдущая попытка не прошла проверку: {feedback}\nУчти это в новом плане."
    msg = _chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                response_format={"type": "json_object"})
    return json.loads(msg.content).get("steps", [])[:4]


def execute_step(step: str, context: str) -> str:
    """Шаг EXECUTE: ReAct-под-агент выполняет одну подзадачу через python."""
    messages = [
        {"role": "system", "content":
            "Выполни подзадачу анализа данных. Используй инструмент python "
            "(в ОДНОМ вызове загрузи данные и посчитай нужное, обязательно print() "
            "результат). Доступны pd и DATA_PATH. " + SCHEMA +
            " Если код упал — исправь и повтори, НЕ заявляй, что данных нет. "
            "В конце дай краткий вывод с числами."},
        {"role": "user", "content": f"Контекст: {context}\n\nПодзадача: {step}"},
    ]
    for _ in range(5):
        msg = _chat(messages, tools=PY_TOOL, tool_choice="auto")
        if not msg.tool_calls:
            return (msg.content or "").strip()
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                          "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                          for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            code = json.loads(tc.function.arguments or "{}").get("code", "")
            obs = run_python(code)
            print(f"{C['dim']}    └ python → {obs[:200].splitlines()[0] if obs else ''}…{C['reset']}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs[:3000]})
    return "(шаг не завершён за лимит под-итераций)"


def verify(goal: str, findings: str) -> dict:
    """Шаг VERIFY: проверить, достигнута ли цель и всё ли опирается на данные."""
    sys_p = ('Ты критик-проверяющий. Оцени, достигнута ли цель и подкреплены ли '
             'выводы реальными числами. Верни JSON {"ok": true/false, '
             '"reason": "...", "answer": "итоговый ответ, если ok"}.')
    msg = _chat([{"role": "system", "content": sys_p},
                 {"role": "user", "content": f"Цель: {goal}\n\nНайдено:\n{findings}"}],
                response_format={"type": "json_object"})
    return json.loads(msg.content)


def claw(goal: str, max_iters: int = 2) -> str:
    feedback, findings = "", ""
    for it in range(1, max_iters + 1):
        steps = plan(goal, feedback)
        print(f"{C['plan']}🧭 ПЛАН (итерация {it}):{C['reset']}")
        for i, s in enumerate(steps, 1):
            print(f"   {i}. {s}")
        results = []
        for i, s in enumerate(steps, 1):
            print(f"{C['exec']}⚙️  ШАГ {i}: {s}{C['reset']}")
            r = execute_step(s, findings)
            print(f"   → {r[:300]}")
            results.append(f"[{s}] {r}")
        findings = "\n".join(results)
        v = verify(goal, findings)
        print(f"{C['verify']}🔎 ПРОВЕРКА: ok={v.get('ok')} — {v.get('reason','')}{C['reset']}\n")
        if v.get("ok"):
            return v.get("answer") or findings
        feedback = v.get("reason", "")
    return "Цель не достигнута за отведённые итерации. Последние находки:\n" + findings


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "Разберись, почему в одном из месяцев заметно просела выручка, и дай рекомендацию."
    if not API_KEY:
        print("⚠️  Нет AI_PROVIDER_TOKEN — ключ LLM-роутера выдаёт LMS."); sys.exit(1)
    print(f"🎯 Цель: {goal}\n{'═'*60}")
    print("═" * 60 + "\nИТОГ:\n" + claw(goal))
