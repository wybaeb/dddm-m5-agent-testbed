#!/usr/bin/env python3
"""
Урок 5.2 — SGR как УНИВЕРСАЛЬНАЯ агентная модель: «Next Step» / adaptive planning.

Простой SGR (см. рядом `sgr_pipeline.py`) — один проход по жёсткой схеме: модель
заполняет поля, мы исполняем одно выражение. Предсказуемо, но негибко: один шаг,
никакой адаптации к тому, что вскрылось в данных.

Здесь — тот же принцип SGR (модель на КАЖДОМ шаге обязана заполнить схему рассуждения
перед вызовом инструмента и оценить, завершена ли задача), но развёрнутый в
АГЕНТНЫЙ ЦИКЛ с адаптивным планированием по модели «Next Step» Р. Абдуллина
(https://abdullin.com/schema-guided-reasoning/adaptive-planning):

    на каждом шаге модель возвращает объект NextStep:
      reasoning                  — детальное рассуждение ПЕРЕД действием (ядро SGR)
      current_state              — где мы сейчас
      plan_remaining_steps_brief — 1..5 шагов, что осталось (план пересобирается заново)
      task_completed             — булев флаг «цель достигнута»
      function                   — какой инструмент вызвать ПЕРВЫМ (только он и исполнится)

    Мы исполняем только первый шаг, дописываем наблюдение в диалог и просим модель
    спланировать заново. Старый план не храним — на каждом витке он создаётся с нуля
    с учётом того, что уже вскрылось в данных. Это и есть adaptive planning.

── Почему это важно именно для данных ──────────────────────────────────────────
Большой DataFrame (сотни тысяч строк) НИКОГДА не попадает в контекстное окно модели.
Он живёт в серверной сессии (`DataSession`). Модель видит только:
  • схему (имена/типы столбцов) и число строк — но не сами данные;
  • СЭМПЛ ≤ SAMPLE_CAP строк по явному запросу (жёсткий потолок на стороне кода);
  • короткую СВОДКУ результата после каждого преобразования (форма + маленький head).
Инструменты работают над полным датафреймом, а через контекст ходят только сэмплы и
сводки. Так модель планирует следующий шаг, не «проглатывая» весь массив.

Инструменты намеренно ограничены (принцип SGR — сузить меню действий):
  describe_data — столбцы, типы, число строк (без данных)
  show_sample   — до SAMPLE_CAP строк текущего рабочего кадра
  transform     — одно pandas-выражение над df (база) / cur (текущий результат);
                  результат становится новым `cur`; в ответ уходит только сводка
  finish        — завершить с деловым выводом

Запуск:
    export AI_PROVIDER_TOKEN=...
    python3 sgr_next_step.py "Найди канал с худшим средним чеком и проверь, не тянет ли его один город"
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

SAMPLE_CAP = 10      # максимум строк, которые модель может УВИДЕТЬ за раз
MAX_STEPS = 8        # потолок на число вызовов инструментов (SGR: ограничиваем цикл)
OBS_CHARS = 1200     # обрезка наблюдения — в контекст не уходит «простыня» данных

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ── Серверная сессия данных: большой df живёт ЗДЕСЬ, не в контексте модели ──────
class DataSession:
    def __init__(self, df: pd.DataFrame):
        self.df = df          # неизменяемая база
        self.cur = df         # текущий рабочий результат (то, над чем идём дальше)

    def describe(self) -> str:
        cols = ", ".join(f"{c}:{t}" for c, t in self.df.dtypes.astype(str).items())
        return f"строк: {len(self.df)}; столбцы: {cols}"

    def sample(self, n: int, columns=None) -> str:
        n = max(1, min(int(n), SAMPLE_CAP))            # жёсткий потолок сэмпла
        obj = self.cur
        if columns and isinstance(obj, pd.DataFrame):
            keep = [c for c in columns if c in obj.columns]
            if keep:
                obj = obj[keep]
        head = obj.head(n) if hasattr(obj, "head") else obj
        note = "" if n >= _len(self.cur) else f"  (показаны {n} из {_len(self.cur)} — это сэмпл)"
        return f"sample[{n}]{note}:\n{head.to_string() if hasattr(head,'to_string') else head}"

    def transform(self, expr: str) -> str:
        _guard(expr)
        res = eval(expr, {"df": self.df, "cur": self.cur, "pd": pd})  # noqa: S307
        self.cur = res
        return _summary(res)


def _len(obj) -> int:
    try:
        return len(obj)
    except TypeError:
        return 1


def _guard(expr: str):
    # eval() синтаксически не допускает присваиваний-стейтментов, поэтому фильтры
    # (df[df.channel == 'X']) и kwargs (agg(x=...)) безопасны — блокируем только
    # реально опасное: импорты, дандеры, ввод-вывод, побег из песочницы.
    banned = ("import", "__", "open(", "exec", "eval", "os.", "sys.",
              "subprocess", "system", "to_csv", "to_pickle", "to_parquet", "read_")
    if any(t in expr for t in banned):
        raise ValueError("Выражение нарушает ограничения SGR (только чтение/агрегация над df/cur).")


def _summary(res) -> str:
    """Короткая СВОДКА результата — не сами данные. Именно это уходит в контекст."""
    if isinstance(res, pd.DataFrame):
        body = res.head(5).to_string()
        return f"DataFrame {res.shape[0]}×{res.shape[1]}; head(5):\n{body}"
    if isinstance(res, pd.Series):
        body = res.head(5).to_string()
        return f"Series[{len(res)}]; head(5):\n{body}"
    return f"скаляр: {res}"


# ── Меню инструментов (сужено намеренно) ───────────────────────────────────────
def run_tool(sess: DataSession, name: str, args: dict) -> str:
    if name == "describe_data":
        return sess.describe()
    if name == "show_sample":
        return sess.sample(args.get("n", SAMPLE_CAP), args.get("columns"))
    if name == "transform":
        return sess.transform(args.get("pandas_expr", ""))
    if name == "finish":
        return "OK"
    return f"неизвестный инструмент: {name}"


TOOLS_DOC = (
    "Доступные инструменты (function.name / arguments):\n"
    "  describe_data {}                      — столбцы, типы, число строк (без данных)\n"
    f"  show_sample   {{n<= {SAMPLE_CAP}, columns?}}      — до {SAMPLE_CAP} строк текущего кадра cur\n"
    "  transform     {pandas_expr}           — ОДНО выражение над df (база) или cur (текущий результат),\n"
    "                                          напр. df.groupby('channel').revenue.mean().sort_values()\n"
    "  finish        {answer}                — завершить: answer = деловой вывод 2-4 предложения\n"
)

NEXTSTEP_DOC = (
    "Верни СТРОГО JSON-объект NextStep ровно с ключами:\n"
    '  "reasoning": строка — детальное рассуждение ПЕРЕД действием;\n'
    '  "current_state": строка — где ты сейчас, что уже узнал;\n'
    '  "plan_remaining_steps_brief": массив из 1..5 коротких строк — что осталось;\n'
    '  "task_completed": bool — достигнута ли цель;\n'
    '  "function": {"name": строка, "arguments": объект} — ПЕРВЫЙ шаг из плана.\n'
    "Исполнится только function. На следующем витке спланируешь заново. "
    "Когда ответ готов — вызови finish с полем answer и поставь task_completed=true.\n"
    "ВАЖНО: весь датафрейм тебе НЕ показывают. Ориентируйся на describe_data, "
    f"сэмплы (<= {SAMPLE_CAP} строк) и сводки transform. Каждое число получай через transform."
)


def next_step(question: str, history: list) -> dict:
    sys_p = ("Ты аналитик-агент по модели Schema-Guided Reasoning (Next Step / adaptive planning). "
             "На каждом шаге сначала рассуждаешь, потом выбираешь ОДИН инструмент.\n"
             + TOOLS_DOC + "\n" + NEXTSTEP_DOC)
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": f"Цель: {question}"}]
    msgs += history
    raw = client.chat.completions.create(
        model=MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=msgs,
    ).choices[0].message.content
    return json.loads(raw)


def render(step: dict):
    print("📋 NEXTSTEP:")
    print(f"   reasoning : {step.get('reasoning','')}")
    print(f"   state     : {step.get('current_state','')}")
    plan = step.get("plan_remaining_steps_brief", [])
    print("   plan      : " + " → ".join(map(str, plan)) if plan else "   plan      : —")
    fn = step.get("function", {}) or {}
    print(f"   completed : {step.get('task_completed')}")
    print(f"   call      : {fn.get('name')} {json.dumps(fn.get('arguments', {}), ensure_ascii=False)}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or \
        "Найди канал с худшим средним чеком и проверь, не тянет ли его один город"
    if not API_KEY:
        print("⚠️  Нет AI_PROVIDER_TOKEN — ключ LLM-роутера выдаёт LMS."); sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    sess = DataSession(df)

    print(f"❓ {question}\n{'─'*64}")
    print(f"ℹ️  df: {len(df)} строк — в контекст модели он НЕ передаётся, "
          f"только сэмплы (<= {SAMPLE_CAP}) и сводки.\n{'─'*64}")

    history, answer = [], None
    for i in range(1, MAX_STEPS + 1):
        print(f"— Шаг {i} " + "─" * 54)
        step = next_step(question, history)
        render(step)
        fn = step.get("function", {}) or {}
        name, args = fn.get("name", ""), fn.get("arguments", {}) or {}

        if name == "finish" or step.get("task_completed"):
            answer = args.get("answer") or step.get("current_state", "")
            print("─" * 64)
            print("🟣 ВЫВОД:", answer)
            break

        try:
            obs = run_tool(sess, name, args)
        except Exception as e:                      # noqa: BLE001 — показать модели её ошибку
            obs = f"ОШИБКА инструмента: {e}"
        obs = obs[:OBS_CHARS] + (" …" if len(obs) > OBS_CHARS else "")
        print(f"👁  НАБЛЮДЕНИЕ:\n{obs}")

        # В историю кладём КОМПАКТНО: решение модели + наблюдение (без больших данных)
        history.append({"role": "assistant",
                        "content": json.dumps({"function": fn}, ensure_ascii=False)})
        history.append({"role": "user",
                        "content": f"Наблюдение по инструменту {name}:\n{obs}\nПланируй следующий шаг."})
    else:
        print("─" * 64)
        print("⚠️  Достигнут потолок шагов (MAX_STEPS) — цикл ограничен намеренно (SGR).")
