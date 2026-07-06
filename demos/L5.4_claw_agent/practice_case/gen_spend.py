#!/usr/bin/env python3
"""
Генератор второго (нарочно «грязного») источника для практического кейса 5.4.

Делает `marketing_spend.csv` — помесячные расходы на маркетинг по каналам за 2024.
Файл специально неаккуратный, как выгрузка из реальной рекламной системы:
  • даты в трёх форматах вперемешку: «2024-01», «Jan 2024», «01.2024»;
  • названия каналов с вариантами написания: «Соц.сети», «органика », «E-mail»;
  • суммы с ₽ и пробелами-разделителями: «1 200 000 ₽», иногда просто число;
  • один (канал, месяц) пропущен; одна строка-дубль;
  • один месяц по «Реклама» — аномальный всплеск расхода (для поиска выброса).

Агент-харнесс должен сам ПЕРЕРАБОТАТЬ эти данные кодом, прежде чем считать.
Детерминированно (без random) — у всех одинаковый файл.
"""
import csv
import os

CHANNELS = ["Реклама", "Соцсети", "Email", "Органика", "Реферал"]
# базовый месячный расход по каналу, ₽
BASE = {"Реклама": 3_000_000, "Соцсети": 1_800_000, "Email": 300_000,
        "Органика": 200_000, "Реферал": 500_000}
# сезонный множитель по месяцу (1..12): лето тише, декабрь — пик распродаж
SEASON = {1: 1.0, 2: 0.95, 3: 1.05, 4: 1.0, 5: 0.9, 6: 0.8,
          7: 0.8, 8: 0.85, 9: 1.05, 10: 1.15, 11: 1.3, 12: 1.45}

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# варианты «грязного» написания каналов — подменяем по детерминированному правилу
DIRTY_NAME = {"Соцсети": "Соц.сети", "Органика": "органика ", "Email": "E-mail"}


def fmt_month(month: int, variant: int) -> str:
    """Три формата даты вперемешку — variant=0/1/2."""
    if variant == 0:
        return f"2024-{month:02d}"
    if variant == 1:
        return f"{MONTHS_EN[month - 1]} 2024"
    return f"{month:02d}.2024"


def fmt_spend(value: int, variant: int) -> str:
    """Суммы тоже неаккуратные: с ₽ и пробелами либо просто число."""
    if variant == 0:
        s = f"{value:,}".replace(",", " ")
        return f"{s} ₽"
    if variant == 1:
        return str(value)
    return f"{value:.1f}"


def main():
    out = os.path.join(os.path.dirname(__file__), "marketing_spend.csv")
    rows = []
    for ci, ch in enumerate(CHANNELS):
        for m in range(1, 13):
            # пропускаем один (канал, месяц): Email за август — «дырка» в выгрузке
            if ch == "Email" and m == 8:
                continue
            spend = int(BASE[ch] * SEASON[m])
            # аномальный всплеск: Реклама в ноябре (перегрели бюджет под распродажу)
            if ch == "Реклама" and m == 11:
                spend = int(spend * 3.0)
            variant = (ci + m) % 3
            name = DIRTY_NAME[ch] if (ci + m) % 4 == 0 and ch in DIRTY_NAME else ch
            rows.append([fmt_month(m, variant), name, fmt_spend(spend, variant)])
    # одна строка-дубль (Реферал, март) — частая беда выгрузок
    dup = [r for r in rows if r[0] in ("2024-03", "Mar 2024", "03.2024") and "еферал" in r[1].lower()]
    if dup:
        rows.insert(rows.index(dup[0]) + 1, list(dup[0]))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "channel", "spend"])
        w.writerows(rows)
    print(f"Записал {len(rows)} строк → {out}")


if __name__ == "__main__":
    main()
