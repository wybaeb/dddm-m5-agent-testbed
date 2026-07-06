#!/usr/bin/env python3
"""
Бот-воркспейс студента (Модуль 5, практика).

Каждый студент:
  1) пишет боту /start → вводит логин и пароль (выданы в LMS);
  2) после авторизации получает СВОЙ изолированный рабочий каталог;
  3) общается с персональным ReAct-агентом обычными сообщениями:
     «построй график выручки по месяцам», «найди топ-5 городов» и т.п.;
  4) видит прозрачный лог агента (Мысль/Действие/Наблюдение) и получает
     графики картинками прямо в чат.

Транспорт — long polling (getUpdates), как у настоящих ботов без публичного
URL. Реализован на чистом httpx, без тяжёлых фреймворков — чтобы был виден сам
протокол. Тот же поллинг работает и для бота в Max (мессенджер): меняется
только базовый URL и имена методов API (см. класс Transport).

Запуск:
    cp users.example.json users.json   # завести студентов
    export BOT_TOKEN=...                # токен Telegram-бота от @BotFather
    export AI_PROVIDER_TOKEN=...        # ключ LLM-роутера (LMS)
    python3 bot.py
"""
import os
import json
import time
import hashlib
import traceback

import httpx

from agent_core import Agent

USERS_FILE = os.environ.get("USERS_FILE", os.path.join(os.path.dirname(__file__), "users.json"))
WORKSPACES = os.environ.get("WORKSPACES_DIR", os.path.join(os.path.dirname(__file__), "workspaces"))
TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


# --------------------------------------------------------------------------- #
#  Транспорт мессенджера (Telegram). Для Max — те же 3 метода, другой base_url.
# --------------------------------------------------------------------------- #
class Transport:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self.http = httpx.Client(timeout=70)

    def get_updates(self, offset: int):
        r = self.http.get(f"{self.base}/getUpdates",
                          params={"offset": offset, "timeout": 50})
        return r.json().get("result", [])

    def send_message(self, chat_id, text: str):
        # Telegram ограничивает 4096 символов на сообщение
        for chunk in _chunks(text, 3500):
            self.http.post(f"{self.base}/sendMessage",
                          json={"chat_id": chat_id, "text": chunk})

    def send_photo(self, chat_id, path: str, caption: str = ""):
        with open(path, "rb") as f:
            self.http.post(f"{self.base}/sendPhoto",
                          data={"chat_id": chat_id, "caption": caption},
                          files={"photo": f})


def _chunks(text: str, n: int):
    text = text or "(пусто)"
    for i in range(0, len(text), n):
        yield text[i:i + n]


# --------------------------------------------------------------------------- #
#  Пользователи и авторизация
# --------------------------------------------------------------------------- #
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        raise SystemExit(f"Нет файла {USERS_FILE} — скопируй users.example.json → users.json")
    return json.load(open(USERS_FILE))


# --------------------------------------------------------------------------- #
#  Бот
# --------------------------------------------------------------------------- #
class WorkspaceBot:
    def __init__(self):
        self.tg = Transport(os.environ["BOT_TOKEN"])
        self.users = load_users()
        self.sessions: dict[int, dict] = {}  # chat_id -> session

    def session(self, chat_id: int) -> dict:
        return self.sessions.setdefault(chat_id, {"stage": "anon", "login": None, "agent": None})

    # --- обработка одного сообщения ---------------------------------------- #
    def handle(self, chat_id: int, text: str):
        s = self.session(chat_id)
        text = (text or "").strip()

        if text in ("/start", "/help"):
            return self.tg.send_message(chat_id, HELP)
        if text == "/logout":
            self.sessions.pop(chat_id, None)
            return self.tg.send_message(chat_id, "Вы вышли. /start — войти снова.")

        # --- авторизация (логин → пароль) --- #
        if s["stage"] != "authed":
            return self._auth_step(chat_id, text)

        # --- авторизованные команды --- #
        if text == "/reset":
            s["agent"].reset()
            return self.tg.send_message(chat_id, "История диалога очищена. Файлы на месте.")
        if text == "/whoami":
            u = self.users[s["login"]]
            return self.tg.send_message(chat_id, f"Вы: {u['name']} (логин {s['login']}). Воркспейс: workspaces/{s['login']}/")
        if text == "/tasks":
            return self.tg.send_message(chat_id, list_tasks())
        if text.startswith("/task"):
            return self.tg.send_message(chat_id, read_task(text))

        # --- обычное сообщение → агент --- #
        return self._run_agent(chat_id, text)

    def _auth_step(self, chat_id: int, text: str):
        s = self.session(chat_id)
        if s["stage"] == "anon":
            if text.startswith("/login"):
                parts = text.split()
                if len(parts) == 3:  # /login <log> <pass> одной строкой
                    return self._try_login(chat_id, parts[1], parts[2])
            s["stage"] = "await_login"
            return self.tg.send_message(chat_id, "Введите логин (выдан в LMS):")
        if s["stage"] == "await_login":
            login = text.strip()
            if login not in self.users:
                return self.tg.send_message(chat_id, "Логин не найден. Попробуйте ещё раз:")
            s["login"], s["stage"] = login, "await_pass"
            return self.tg.send_message(chat_id, "Введите пароль:")
        if s["stage"] == "await_pass":
            return self._try_login(chat_id, s["login"], text.strip())

    def _try_login(self, chat_id: int, login: str, password: str):
        s = self.session(chat_id)
        user = self.users.get(login)
        if not user or user["password_sha256"] != _hash(password):
            s.update(stage="anon", login=None)
            return self.tg.send_message(chat_id, "Неверный логин или пароль. /start — заново.")
        ws = os.path.join(WORKSPACES, login)
        provision_workspace(ws)
        s.update(stage="authed", login=login,
                 agent=Agent(ws, on_event=lambda *a: self._on_event(chat_id, *a)))
        self.tg.send_message(chat_id, f"✅ Здравствуйте, {user['name']}! Ваш персональный аналитический агент готов.\n\n"
                                       "Просто напишите задачу, например: «построй график выручки по месяцам».\n"
                                       "/tasks — список учебных заданий, /help — все команды.")

    # --- запуск агента + трансляция шагов --- #
    def _run_agent(self, chat_id: int, text: str):
        s = self.session(chat_id)
        self.tg.send_message(chat_id, "🤖 Работаю над задачей…")
        try:
            result = s["agent"].ask(text)
        except Exception:  # noqa: BLE001
            return self.tg.send_message(chat_id, "Ошибка агента:\n" + traceback.format_exc()[-1500:])
        for img in result["images"]:
            try:
                self.tg.send_photo(chat_id, img, caption="📊 график")
            except Exception:  # noqa: BLE001
                pass
        self.tg.send_message(chat_id, "🟣 Ответ:\n" + result["answer"])

    def _on_event(self, chat_id: int, kind: str, step: int, payload):
        """Транслировать шаг ReAct в чат (прозрачность агента)."""
        if kind == "thought":
            self.tg.send_message(chat_id, f"💭 [{step}] {payload}")
        elif kind == "action":
            tool = payload["tool"]
            arg = payload["args"].get("code") or payload["args"].get("query") or ""
            body = f"\n{arg[:600]}" if arg else ""
            self.tg.send_message(chat_id, f"⚙️ [{step}] {tool}{body}")
        elif kind == "obs":
            self.tg.send_message(chat_id, f"👁 [{step}] {str(payload)[:600]}")
        # final отправляется отдельно в _run_agent

    # --- главный цикл long polling --- #
    def run(self):
        print("Bot polling… Ctrl+C — стоп.")
        offset = 0
        while True:
            try:
                updates = self.tg.get_updates(offset)
            except Exception as e:  # noqa: BLE001
                print("poll error:", e); time.sleep(3); continue
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg or "text" not in msg:
                    continue
                chat_id = msg["chat"]["id"]
                try:
                    self.handle(chat_id, msg["text"])
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                    try:
                        self.tg.send_message(chat_id, "Внутренняя ошибка, попробуйте ещё раз.")
                    except Exception:  # noqa: BLE001
                        pass


# --------------------------------------------------------------------------- #
#  Воркспейс студента и задания
# --------------------------------------------------------------------------- #
def provision_workspace(ws: str):
    """Создать каталог студента и положить туда учебный датасет (один раз)."""
    os.makedirs(ws, exist_ok=True)
    target = os.path.join(ws, "sales_data.csv")
    if not os.path.exists(target):
        src = os.path.join(os.path.dirname(__file__), "..", "shared", "data", "sales_data.csv")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, target)


def list_tasks() -> str:
    if not os.path.isdir(TASKS_DIR):
        return "Заданий пока нет."
    cards = sorted(f for f in os.listdir(TASKS_DIR) if f.endswith(".md"))
    lines = ["Учебные задания (открыть: /task N):"]
    for f in cards:
        num = f.split("_")[0]
        title = open(os.path.join(TASKS_DIR, f)).readline().lstrip("# ").strip()
        lines.append(f"  /task {num} — {title}")
    return "\n".join(lines)


def read_task(text: str) -> str:
    parts = text.split()
    if len(parts) < 2:
        return "Укажите номер: /task 1"
    num = parts[1].zfill(2)
    for f in os.listdir(TASKS_DIR):
        if f.startswith(num) and f.endswith(".md"):
            return open(os.path.join(TASKS_DIR, f)).read()[:3500]
    return f"Задание {num} не найдено. /tasks — список."


HELP = """🤖 Персональный аналитический агент (Модуль 5)

Это ReAct-агент в вашем личном рабочем каталоге. Он умеет считать, строить
графики и делать выводы по данным — просто опишите задачу словами.

Авторизация:
  /start  — начать, ввести логин и пароль (выданы в LMS)
  /logout — выйти

После входа:
  просто напишите задачу — «построй график выручки по месяцам»
  /tasks      — список учебных заданий
  /task N     — открыть задание N
  /whoami     — кто вы и где ваш воркспейс
  /reset      — очистить историю диалога (файлы остаются)
  /help       — эта справка"""


if __name__ == "__main__":
    if not os.environ.get("BOT_TOKEN"):
        raise SystemExit("Нет BOT_TOKEN — получите токен у @BotFather и экспортируйте.")
    if not os.environ.get("AI_PROVIDER_TOKEN"):
        raise SystemExit("Нет AI_PROVIDER_TOKEN — ключ LLM-роутера выдаёт LMS.")
    WorkspaceBot().run()
