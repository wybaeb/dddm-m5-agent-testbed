#!/usr/bin/env python3
"""
Локальный чат с агентом — БЕЗ Telegram (для отладки и грейдинга).

Тот же Agent, что использует бот, но общение идёт в терминале. Удобно проверить
харнесс, не заводя бота у @BotFather. Графики сохраняются в workspace и путь к
ним печатается.

Запуск:
    export AI_PROVIDER_TOKEN=...
    python3 local_chat.py            # интерактивно
    python3 local_chat.py "вопрос"   # один вопрос и выход
"""
import os
import sys

from agent_core import Agent

WS = os.environ.get("LOCAL_WS", os.path.join(os.path.dirname(__file__), "workspaces", "_local"))

LABELS = {"thought": "💭 Мысль", "action": "⚙️  Действие", "obs": "👁  Наблюдение", "final": "✅ Ответ"}


def on_event(kind, step, payload):
    if kind == "action":
        body = payload["args"].get("code") or payload["args"].get("query") or ""
        print(f"\033[33m[{step}] {LABELS[kind]} · {payload['tool']}\033[0m\n\033[2m{body}\033[0m")
    elif kind == "obs":
        print(f"\033[32m[{step}] {LABELS[kind]}\033[0m\n\033[2m{str(payload)[:1000]}\033[0m")
    else:
        print(f"\033[36m[{step}] {LABELS[kind]}\033[0m\n{payload}\n")


def main():
    # положим учебный датасет рядом
    os.makedirs(WS, exist_ok=True)
    src = os.path.join(os.path.dirname(__file__), "..", "shared", "data", "sales_data.csv")
    dst = os.path.join(WS, "sales_data.csv")
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil; shutil.copy(src, dst)

    agent = Agent(WS, on_event=on_event)
    if len(sys.argv) > 1:
        res = agent.ask(" ".join(sys.argv[1:]))
        if res["images"]:
            print("📊 графики:", ", ".join(res["images"]))
        return
    print("Локальный чат с агентом. Пустая строка — выход.\n")
    while True:
        try:
            q = input("\033[1mВы:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        res = agent.ask(q)
        if res["images"]:
            print("📊 графики:", ", ".join(res["images"]))


if __name__ == "__main__":
    main()
