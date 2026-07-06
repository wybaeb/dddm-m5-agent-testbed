#!/usr/bin/env python3
"""
Урок 5.6 — Claw-like агент поверх MCP-витрины данных.

То же автономное поведение (план → исполнение → проверка), что в L5.4, но
инструменты агент берёт не из локального кода, а из MCP-сервера (корпоративная
витрина данных, опубликованная в Модуле 4). Агент видит только то, что витрина
разрешила: describe_schema и run_sql (read-only). Это «корпоративный контур»:
автономия агента ограничена разъёмом MCP.

Требуется: поднятый MCP-сервер (см. ../L4.1_tool_calling/mcp_server.sh) и БД.
Если их нет — скрипт честно сообщит и покажет, как запустить.

Запуск:
    export AI_PROVIDER_TOKEN=...
    export MCP_URL=http://localhost:7800/mcp
    python3 claw_over_mcp.py "Найди сегмент с просадкой выручки"
"""
import os
import sys
import json
import asyncio

MCP_URL = os.environ.get("MCP_URL", "http://localhost:7800/mcp")
API_KEY = os.environ.get("AI_PROVIDER_TOKEN", "")
BASE_URL = os.environ.get("AI_PROVIDER_URL", "https://api.agentplatform.ru/v1")
MODEL = os.environ.get("AI_PROVIDER_MODEL", "openai/gpt-4o")


async def amain(goal: str):
    try:
        from openai import OpenAI
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        print("Нужны пакеты openai и mcp: pip install openai mcp"); return

    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                oa_tools = [{"type": "function", "function": {
                    "name": t.name, "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}}}}
                    for t in tools]
                print(f"🔌 MCP-витрина на {MCP_URL}: инструменты — {', '.join(t.name for t in tools)}")

                client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
                messages = [
                    {"role": "system", "content":
                        "Ты автономный аналитический агент. Цель достигай через MCP-инструменты "
                        "витрины данных: сначала describe_schema, потом run_sql (только SELECT). "
                        "Действуй циклом план→запрос→проверка, в конце дай вывод с числами."},
                    {"role": "user", "content": goal},
                ]
                for step in range(1, 12):
                    msg = client.chat.completions.create(
                        model=MODEL, messages=messages, tools=oa_tools,
                        tool_choice="auto", temperature=0).choices[0].message
                    if msg.content:
                        print(f"💭 [{step}] {msg.content.strip()[:300]}")
                    if not msg.tool_calls:
                        print("\n✅ ИТОГ:\n" + (msg.content or "")); return
                    messages.append({"role": "assistant", "content": msg.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function", "function": {
                            "name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in msg.tool_calls]})
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments or "{}")
                        print(f"⚙️  [{step}] {tc.function.name}({json.dumps(args, ensure_ascii=False)[:120]})")
                        res = await session.call_tool(tc.function.name, args)
                        out = "\n".join(c.text for c in res.content if hasattr(c, "text"))
                        print(f"👁  {out[:200]}")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:3000]})
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  MCP-витрина недоступна ({e}).")
        print("Подними её: cd ../L4.1_tool_calling && ./mcp_server.sh  (нужна учебная БД).")


if __name__ == "__main__":
    if not API_KEY:
        print("⚠️  Нет AI_PROVIDER_TOKEN — ключ LLM-роутера выдаёт LMS."); sys.exit(1)
    goal = " ".join(sys.argv[1:]) or "Найди сегмент с самой низкой выручкой и предложи гипотезу, что проверить."
    asyncio.run(amain(goal))
