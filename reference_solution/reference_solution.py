#!/usr/bin/env python3
"""
ЭТАЛОННОЕ РЕШЕНИЕ кейса 5.4 — для МЕТОДИСТА (студенту/агенту НЕ выдаётся).

Нужно, чтобы (1) убедиться, что кейс в принципе решаем, и (2) показать «планку»:
какого уровня результат должен выдать Claw-like агент в своей песочнице.

Это ровно тот тип работы, который отличает харнесс-агента от обычного ReAct:
агент сам пишет КОД, который ПЕРЕРАБАТЫВАЕТ два источника (один — грязный),
строит нестандартные таблицы (когортное удержание, экономика каналов) и
СОБИРАЕТ ФАЙЛ-ОТЧЁТ. Один промпт «ответь числом» так не умеет.

Запуск:
    python3 reference_solution.py
Артефакты (рядом со скриптом):
    retention_heatmap.png · report.html · EXECUTIVE_SUMMARY.md
"""
import base64
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SALES = os.path.join(HERE, "..", "..", "shared", "data", "sales_data.csv")
SPEND = os.path.join(HERE, "marketing_spend.csv")

MONTHS_EN = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
# карта нормализации грязных названий каналов → канон
CANON = {"соц.сети": "Соцсети", "соцсети": "Соцсети", "органика": "Органика",
         "e-mail": "Email", "email": "Email", "реклама": "Реклама",
         "реферал": "Реферал"}

reconcile_log = []  # человекочитаемый журнал «что почистили» — это часть отчёта


# ───────────────────────── 1. чистка грязного источника ─────────────────────────
def parse_month(raw: str) -> int:
    s = str(raw).strip()
    if re.fullmatch(r"2024-\d{2}", s):
        return int(s[5:7])
    if re.fullmatch(r"\d{2}\.2024", s):
        return int(s[:2])
    m = re.fullmatch(r"([A-Za-z]{3})\s+2024", s)
    if m:
        return MONTHS_EN[m.group(1).title()]
    raise ValueError(f"неизвестный формат даты: {raw!r}")


def parse_spend(raw: str) -> float:
    s = re.sub(r"[^\d.]", "", str(raw).replace(" ", ""))
    return float(s)


def norm_channel(raw: str) -> str:
    key = str(raw).strip().lower()
    return CANON.get(key, str(raw).strip())


def load_spend() -> pd.DataFrame:
    df = pd.read_csv(SPEND)
    n0 = len(df)
    df["month"] = df["month"].map(parse_month)
    df["channel_raw"] = df["channel"]
    df["channel"] = df["channel"].map(norm_channel)
    renamed = (df["channel_raw"].str.strip() != df["channel"]).sum()
    if renamed:
        reconcile_log.append(f"Нормализовано написаний каналов: {renamed} "
                             f"(напр. «Соц.сети»→«Соцсети», «органика »→«Органика»).")
    df["spend"] = df["spend"].map(parse_spend)
    # дубли (канал, месяц)
    dups = df.duplicated(subset=["month", "channel"]).sum()
    if dups:
        df = df.drop_duplicates(subset=["month", "channel"], keep="first")
        reconcile_log.append(f"Удалено строк-дублей (канал, месяц): {dups}.")
    df = df[["month", "channel", "spend"]]
    # пропущенные (канал, месяц): достраиваем сетку, дыры — медианой канала (с пометкой)
    grid = pd.MultiIndex.from_product(
        [range(1, 13), sorted(df["channel"].unique())], names=["month", "channel"])
    full = df.set_index(["month", "channel"]).reindex(grid)
    missing = full["spend"].isna()
    if missing.any():
        med = df.groupby("channel")["spend"].median()
        for (m, ch) in full.index[missing]:
            full.loc[(m, ch), "spend"] = med[ch]
        gaps = [f"{ch}/мес {m}" for (m, ch) in full.index[missing]]
        reconcile_log.append(f"Заполнено пропусков медианой канала: {len(gaps)} "
                             f"({', '.join(gaps)}).")
    reconcile_log.insert(0, f"Источник расходов: {n0} строк → {len(full)} "
                            f"чистых записей (канал × 12 мес).")
    return full.reset_index()


# ───────────────────────── 2. продажи: когорты и first-touch ─────────────────────────
def load_sales() -> pd.DataFrame:
    df = pd.read_csv(SALES)
    df["date"] = pd.to_datetime(df["date"])
    df["ym"] = df["date"].dt.month  # 2024 целиком → достаточно номера месяца
    return df


def main():
    spend = load_spend()
    sales = load_sales()

    # first-touch: канал и месяц ПЕРВОГО заказа клиента = канал привлечения и когорта
    first = sales.sort_values("date").groupby("customer_id").first()
    first_touch = first[["channel", "ym"]].rename(
        columns={"channel": "acq_channel", "ym": "acq_month"})
    sales = sales.merge(first_touch, on="customer_id")

    # новые клиенты по (канал, месяц привлечения)
    new_cust = (first_touch.reset_index()
                .groupby(["acq_channel", "acq_month"])["customer_id"].nunique()
                .reset_index(name="new_customers"))

    # ── CAC = расход / новые клиенты (канал × месяц) ──
    cac = spend.merge(new_cust, left_on=["channel", "month"],
                      right_on=["acq_channel", "acq_month"], how="left")
    cac["new_customers"] = cac["new_customers"].fillna(0).astype(int)
    cac["CAC"] = np.where(cac["new_customers"] > 0,
                          cac["spend"] / cac["new_customers"], np.nan)

    # ── когортная матрица удержания (когорта × смещение месяца) ──
    active = (sales.groupby(["acq_month", "ym"])["customer_id"].nunique()
              .reset_index(name="active"))
    size = first_touch.groupby("acq_month").size().rename("size")
    active = active.merge(size, on="acq_month")
    active["offset"] = active["ym"] - active["acq_month"]
    active = active[active["offset"] >= 0]
    active["retention"] = active["active"] / active["size"]
    matrix = active.pivot(index="acq_month", columns="offset", values="retention")

    # ── экономика каналов: ROMI = выручка привлечённых / расход ──
    rev_by_ch = sales.groupby("acq_channel")["revenue"].sum().rename("revenue")
    spend_by_ch = spend.groupby("channel")["spend"].sum().rename("spend")
    new_by_ch = new_cust.groupby("acq_channel")["new_customers"].sum().rename("new")
    econ = pd.concat([spend_by_ch, rev_by_ch, new_by_ch], axis=1).fillna(0)
    econ["CAC"] = econ["spend"] / econ["new"]
    econ["LTV_to_date"] = econ["revenue"] / econ["new"]
    econ["ROMI"] = econ["revenue"] / econ["spend"]
    econ = econ.sort_values("ROMI", ascending=False)

    # ── аномалии расхода: для каждого канала ищем месяцы-выбросы по ЕГО ЖЕ
    #    распределению (IQR 1.5×). Метод, не кейс: ловит перегрев бюджета. ──
    rows = []
    for ch, g in spend.groupby("channel"):
        s = g["spend"]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        hi = q3 + 1.5 * (q3 - q1)
        for r in g[g["spend"] > hi].itertuples():
            rows.append({"channel": ch, "month": int(r.month), "spend": r.spend,
                         "typical": s.median(), "x": r.spend / s.median()})
    outliers = pd.DataFrame(rows).sort_values("spend", ascending=False) if rows \
        else pd.DataFrame(columns=["channel", "month", "spend", "typical", "x"])

    # ── когорты с резким падением удержания (offset 1 < половины медианы) ──
    if 1 in matrix.columns:
        m1 = matrix[1].dropna()
        weak = m1[m1 < m1.median() * 0.6]
    else:
        weak = pd.Series(dtype=float)

    _render(matrix, econ, cac, outliers, weak)


# ───────────────────────── 3. отчёт-файл (нестандартный артефакт) ─────────────────────────
def _heatmap(matrix: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = matrix.reindex(sorted(matrix.index)).values * 100
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd_r", vmin=0, vmax=100)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"+{c}" for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"мес {i}" for i in sorted(matrix.index)])
    ax.set_xlabel("месяцев с привлечения")
    ax.set_ylabel("когорта (месяц привлечения)")
    ax.set_title("Удержание когорт, % активных")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    png = os.path.join(HERE, "retention_heatmap.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def _render(matrix, econ, cac, outliers, weak):
    png = _heatmap(matrix)
    b64 = base64.b64encode(open(png, "rb").read()).decode()

    best, worst = econ.index[0], econ.index[-1]
    summary = f"""# Итоговая записка: окупаемость каналов и удержание когорт

**Самый окупаемый канал:** {best} — ROMI {econ.loc[best,'ROMI']:.1f}×, CAC {econ.loc[best,'CAC']:,.0f} ₽.
**Самый дорогой/слабый:** {worst} — ROMI {econ.loc[worst,'ROMI']:.1f}×, CAC {econ.loc[worst,'CAC']:,.0f} ₽.

**Аномалии расхода (IQR-выброс по каналу):** {'нет' if outliers.empty else ', '.join(f"{r.channel}/мес {int(r.month)} = {r.spend:,.0f} ₽ (×{r.x:.1f} к норме)" for r in outliers.itertuples())}.
**Когорты с просадкой удержания:** {'нет' if weak.empty else ', '.join(f"мес {int(i)}" for i in weak.index)}.

**Рекомендация:** перераспределить часть бюджета с «{worst}» на «{best}»; разобрать
всплеск расхода-выброса (перегрев бюджета без роста привлечения) и причину слабого
удержания ранних когорт.

_Сформировано эталонным решением кейса 5.4 (методическая планка)._
""".replace(",", " ")
    open(os.path.join(HERE, "EXECUTIVE_SUMMARY.md"), "w", encoding="utf-8").write(summary)

    def tbl(df):
        return df.to_html(float_format=lambda x: f"{x:,.1f}".replace(",", " "),
                          border=0, classes="t")

    log_html = "".join(f"<li>{x}</li>" for x in reconcile_log)
    out_html = "<tr><td colspan=4>выбросов не найдено</td></tr>" if outliers.empty else "".join(
        f"<tr><td>{r.channel}</td><td>мес {int(r.month)}</td>"
        f"<td>{r.spend:,.0f} ₽</td><td>×{r.x:.1f}</td></tr>".replace(",", " ")
        for r in outliers.itertuples())

    html = f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Отчёт: каналы и удержание</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;margin:32px auto;color:#1c2430;padding:0 16px}}
h1{{color:#ff5533}} h2{{margin-top:32px;border-bottom:2px solid #ff5533;padding-bottom:4px}}
table.t{{border-collapse:collapse;width:100%;font-size:14px}}
table.t td,table.t th{{border:1px solid #dfe3e8;padding:6px 10px;text-align:right}}
table.t th{{background:#fbece8;text-align:center}}
ul li{{margin:4px 0}} img{{width:100%;border:1px solid #dfe3e8;border-radius:8px}}
.note{{color:#667;font-size:13px}}
</style>
<h1>Окупаемость каналов и удержание когорт</h1>
<p class=note>Кросс-источниковый отчёт: продажи × расходы на маркетинг. Сгенерирован кодом из двух источников (один — «грязный»).</p>
<h2>1. Журнал сверки (что почистили в источнике расходов)</h2>
<ul>{log_html}</ul>
<h2>2. Экономика каналов (CAC · LTV · ROMI)</h2>
{tbl(econ[['spend','new','CAC','LTV_to_date','ROMI']].round(1))}
<h2>3. Удержание когорт</h2>
<img src="data:image/png;base64,{b64}">
<h2>4. Аномалии расхода (IQR по каналу)</h2>
<table class=t><tr><th>Канал</th><th>Месяц</th><th>Расход</th><th>Кратно норме</th></tr>{out_html}</table>
<p class=note>Метод поиска выбросов — IQR (1.5×), одинаково честен на любых данных.</p>
</html>"""
    open(os.path.join(HERE, "report.html"), "w", encoding="utf-8").write(html)

    print("Журнал сверки:")
    for x in reconcile_log:
        print("  •", x)
    print("\nЭкономика каналов (по ROMI):")
    print(econ[["spend", "new", "CAC", "LTV_to_date", "ROMI"]].round(2).to_string())
    print(f"\nАномалии расхода (IQR): {len(outliers)} | слабые когорты: {len(weak)}")
    print("\nАртефакты: retention_heatmap.png · report.html · EXECUTIVE_SUMMARY.md")


if __name__ == "__main__":
    main()
