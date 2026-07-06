#!/usr/bin/env python3
"""
Урок 5.1 / 5.3 — ReAct-агент: цикл «Мысль → Действие → Наблюдение».

Это самый прозрачный вид агента: на каждом шаге модель вслух рассуждает
(Thought), выбирает инструмент (Action), получает результат (Observation)
и повторяет цикл, пока не будет готова дать финальный ответ (Final Answer).

Прозрачность — главное достоинство ReAct для бизнеса: виден КАЖДЫЙ шаг,
любой вывод можно проверить и воспроизвести. Здесь агент анализирует CSV
с продажами, имея ровно один инструмент — выполнение Python-кода.

Запуск:
    export AI_PROVIDER_TOKEN=...        # ключ LLM-роутера (выдаёт LMS)
    python3 react_agent.py "Какие категории товаров растут быстрее всего?"

Зависит только от openai (клиент к OpenAI-совместимому роутеру) и pandas.
"""
import io
import os
import sys
import json
import contextlib
import traceback

from openai import OpenAI

# --- Подключение к LLM-роутеру курса (OpenAI-совместимый) -------------------
BASE_URL = os.environ.get("AI_PROVIDER_URL", "https://api.agentplatform.ru/v1")
API_KEY = os.environ.get("AI_PROVIDER_TOKEN", "")
MODEL = os.environ.get("AI_PROVIDER_MODEL", "openai/gpt-4o")

DATA_PATH = os.environ.get("REACT_DATA", os.path.join(os.path.dirname(__file__), "..", "shared", "data", "sales_data.csv"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = f"""Ты аналитический ReAct-агент. Твоя задача — ответить на вопрос
пользователя о данных, рассуждая пошагово.

У тебя есть ОДИН инструмент: python — выполняет Python-код и возвращает то, что
напечатано через print(). В окружении уже доступны pandas as pd и переменная
DATA_PATH = "{DATA_PATH}" (путь к CSV с продажами). Состояние между вызовами НЕ
сохраняется — каждый вызов это новый процесс, всегда загружай данные заново.
Столбцы CSV: order_id, date (YYYY-MM-DD), customer_id, city, channel,
product_category, product, quantity, price, revenue. Не угадывай имена колонок.

Работай строго по циклу ReAct:
  • сначала коротко опиши мысль (что собираешься выяснить и зачем);
  • затем вызови инструмент python с конкретным кодом;
  • изучи наблюдение (вывод кода);
  • повторяй, пока не соберёшь достаточно фактов.
Когда ответ готов — НЕ вызывай инструмент, а напиши финальный ответ простыми
словами, опираясь на числа, которые реально увидел в наблюдениях.
Не выдумывай цифры: каждое число в ответе должно быть посчитано кодом."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Выполнить Python-код и вернуть вывод print(). Доступны pandas as pd и DATA_PATH.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python-код для выполнения"},
            },
            "required": ["code"],
        },
    },
}]


def run_python(code: str) -> str:
    """Выполнить код в одном пространстве имён, вернуть stdout (или traceback)."""
    import pandas as pd
    ns = {"pd": pd, "DATA_PATH": DATA_PATH}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        return buf.getvalue() + "\n" + traceback.format_exc()
    out = buf.getvalue().strip()
    return out if out else "(код выполнен, но ничего не напечатано — добавь print())"


# Цвета для наглядного лога в терминале
C = {"thought": "\033[36m", "action": "\033[33m", "obs": "\033[32m",
     "final": "\033[1;35m", "reset": "\033[0m", "dim": "\033[2m"}


def react_loop(question: str, max_steps: int = 8, log=print) -> str:
    """Основной цикл ReAct. Возвращает финальный ответ.

    log — функция вывода (по умолчанию print в терминал); бот подменяет её,
    чтобы транслировать те же шаги «Мысль/Действие/Наблюдение» в чат.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=0,
        )
        msg = resp.choices[0].message

        # «Мысль» — это текст, который модель пишет перед/вместо вызова инструмента
        if msg.content and msg.content.strip():
            log("thought", step, msg.content.strip())

        if not msg.tool_calls:
            # Инструмент не вызван → это финальный ответ
            log("final", step, msg.content.strip())
            return msg.content.strip()

        # Зафиксировать намерение модели в истории
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        # Выполнить каждый вызванный инструмент → вернуть наблюдение
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            code = args.get("code", "")
            log("action", step, code)
            observation = run_python(code)
            log("obs", step, observation)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": observation[:4000],  # ограничиваем контекст
            })

    return "Достигнут лимит шагов — агент не успел дойти до ответа."


def _terminal_log(kind: str, step: int, text: str):
    labels = {"thought": "💭 Мысль", "action": "⚙️  Действие (python)",
              "obs": "👁  Наблюдение", "final": "✅ Финальный ответ"}
    head = f"{C[kind]}[{step}] {labels[kind]}{C['reset']}"
    if kind == "action":
        print(f"{head}\n{C['dim']}{text}{C['reset']}\n")
    elif kind == "obs":
        snippet = text if len(text) < 1200 else text[:1200] + " …(обрезано)"
        print(f"{head}\n{C['dim']}{snippet}{C['reset']}\n")
    else:
        print(f"{head}\n{text}\n")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Какие категории товаров приносят больше всего выручки и какие растут быстрее всего по месяцам?"
    if not API_KEY:
        print("⚠️  Не задан AI_PROVIDER_TOKEN — экспортируй ключ LLM-роутера (его выдаёт LMS).")
        sys.exit(1)
    print(f"❓ Вопрос: {question}\n{'─'*60}")
    answer = react_loop(question, log=_terminal_log)
    print("─" * 60)
    print("ИТОГ:", answer)
