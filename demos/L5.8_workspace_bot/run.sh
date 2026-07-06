#!/usr/bin/env bash
# Запуск бота-воркспейса. Перед первым запуском:
#   cp users.example.json users.json   # завести логины/пароли студентов
#   cp .env.example .env                # вписать ключи
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a
[ -f users.json ] || { echo "Нет users.json — cp users.example.json users.json"; exit 1; }

python3 -c "import openai, httpx, pandas, matplotlib" 2>/dev/null || \
  pip install -r requirements.txt

exec python3 bot.py
