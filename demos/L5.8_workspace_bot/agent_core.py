#!/usr/bin/env python3
"""
Ядро ReAct-агента для бота-воркспейса (Модуль 5).

Это упрощённая версия того же харнесса, в котором работают «взрослые»
кодинг-агенты (Claude Code, Cursor, Cline): цикл «Мысль → Действие →
Наблюдение» поверх function calling. Мы намеренно оставили только суть, чтобы
было видно концепцию, а не утонуть в инфраструктуре.

Один экземпляр Agent = одна сессия одного студента: своя история диалога,
свой рабочий каталог (workspace), своя изоляция. Бот создаёт по агенту на
авторизованного студента (см. bot.py).

Логирование шагов вынесено в callback `on_event` — терминал печатает их в
консоль, а бот пересылает те же «Мысль/Действие/Наблюдение» студенту в чат,
чтобы агент оставался прозрачным (главное требование ReAct для бизнеса).
"""
import os
import json

from openai import OpenAI

from sandbox import build_tools, dispatch_tool

BASE_URL = os.environ.get("AI_PROVIDER_URL", "https://api.agentplatform.ru/v1")
API_KEY = os.environ.get("AI_PROVIDER_TOKEN", "")
MODEL = os.environ.get("AI_PROVIDER_MODEL", "openai/gpt-4o")

SYSTEM_PROMPT = """Ты — персональный аналитический агент студента курса DDDM.
Ты работаешь в его личном рабочем каталоге (workspace) и помогаешь анализировать
данные: считаешь, строишь графики, формулируешь выводы.

Архитектура твоей работы — ReAct:
  • Мысль: коротко поясни, что собираешься сделать и зачем.
  • Действие: вызови инструмент (выполнить Python и/или SQL).
  • Наблюдение: изучи результат.
  • Повторяй цикл, пока не соберёшь факты, затем дай финальный ответ словами.

Правила:
  • Каждое число в ответе должно быть РЕАЛЬНО посчитано инструментом — не выдумывай.
  • Состояние Python между вызовами НЕ сохраняется: всегда загружай данные заново.
  • Не дроби на микрошаги: в ОДНОМ вызове python и загрузи данные, и посчитай нужное,
    и при необходимости построй график — так лог остаётся коротким и читаемым.
  • Файлы с данными лежат в текущем каталоге; команда list_files покажет их.
  • Учебный датасет sales_data.csv имеет столбцы: order_id, date (YYYY-MM-DD),
    customer_id, city, channel, product_category, product, quantity, price, revenue.
    Не угадывай имена колонок; для дат делай pd.to_datetime(df['date']).
  • Если код упал — исправь и повтори; не заявляй, что данных нет.
  • Чтобы показать график — построй его в matplotlib и СОХРАНИ в png через
    plt.savefig('имя.png'); бот сам отправит картинку студенту. Не пытайся вывести
    график в stdout.
  • Пиши код чисто, с короткими комментариями по-русски.
  • Финальный ответ — деловой, по существу, с конкретными числами и рекомендацией."""


class Agent:
    """ReAct-агент одного студента."""

    def __init__(self, workspace: str, on_event=None, max_steps: int = 10):
        self.workspace = workspace
        self.on_event = on_event or (lambda *a, **k: None)
        self.max_steps = max_steps
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.tools = build_tools()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        os.makedirs(workspace, exist_ok=True)

    def reset(self):
        """Очистить историю диалога (но не файлы воркспейса)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, user_text: str) -> dict:
        """Прогнать один запрос студента через ReAct-цикл.

        Возвращает {"answer": str, "images": [пути к новым png]}.
        Шаги транслируются через on_event(kind, step, payload).
        """
        self.messages.append({"role": "user", "content": user_text})
        images: list[str] = []

        for step in range(1, self.max_steps + 1):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=self.messages, tools=self.tools,
                tool_choice="auto", temperature=0,
            )
            msg = resp.choices[0].message

            if msg.content and msg.content.strip():
                self.on_event("thought", step, msg.content.strip())

            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                self.messages.append({"role": "assistant", "content": answer})
                self.on_event("final", step, answer)
                return {"answer": answer, "images": images}

            self.messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                self.on_event("action", step, {"tool": name, "args": args})
                observation, new_images = dispatch_tool(name, args, self.workspace)
                images.extend(new_images)
                self.on_event("obs", step, observation)
                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": observation[:4000],
                })

        msg = "Достигнут лимит шагов — уточни вопрос или разбей задачу на части."
        self.on_event("final", self.max_steps, msg)
        return {"answer": msg, "images": images}
