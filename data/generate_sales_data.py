#!/usr/bin/env python3
"""
Generate realistic e-commerce sales dataset for DDDM Module 2 demos.
Used across all lessons: Excel, Python, EDA, SQL, financial modeling.

Output: sales_data.csv (~50,000 rows)
Columns: order_id, date, customer_id, city, channel, product_category, product, quantity, price, revenue
"""
import random
import csv
from datetime import datetime, timedelta

random.seed(42)

CITIES = {
    "Москва": 0.35, "Санкт-Петербург": 0.18, "Казань": 0.08,
    "Новосибирск": 0.07, "Екатеринбург": 0.06, "Нижний Новгород": 0.05,
    "Краснодар": 0.05, "Самара": 0.04, "Ростов-на-Дону": 0.04,
    "Воронеж": 0.03, "Челябинск": 0.03, "Пермь": 0.02,
}

CHANNELS = {"Органика": 0.30, "Реклама": 0.35, "Email": 0.15,
            "Соцсети": 0.12, "Реферал": 0.08}

CATEGORIES = {
    "Электроника": {"products": ["Наушники", "Зарядка", "Кабель USB", "Чехол", "Колонка"],
                     "price_range": (500, 8000), "weight": 0.25},
    "Одежда": {"products": ["Футболка", "Джинсы", "Куртка", "Кроссовки", "Рюкзак"],
               "price_range": (800, 12000), "weight": 0.22},
    "Книги": {"products": ["Бизнес-книга", "Учебник", "Роман", "Нон-фикшн", "Комикс"],
              "price_range": (300, 2500), "weight": 0.15},
    "Дом и сад": {"products": ["Лампа", "Органайзер", "Подушка", "Кашпо", "Полка"],
                   "price_range": (400, 6000), "weight": 0.13},
    "Спорт": {"products": ["Коврик", "Гантели", "Бутылка", "Скакалка", "Эспандер"],
              "price_range": (300, 5000), "weight": 0.10},
    "Красота": {"products": ["Крем", "Шампунь", "Сыворотка", "Маска", "Набор"],
                "price_range": (200, 4000), "weight": 0.08},
    "Продукты": {"products": ["Кофе", "Чай", "Снеки", "Протеин", "Суперфуд"],
                 "price_range": (150, 3000), "weight": 0.07},
}

def weighted_choice(d):
    items = list(d.keys())
    weights = list(d.values())
    return random.choices(items, weights=weights, k=1)[0]

def generate():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    n_days = (end_date - start_date).days + 1

    # Generate ~2000 unique customers
    n_customers = 2000
    customer_cities = {}
    for cid in range(1, n_customers + 1):
        customer_cities[cid] = weighted_choice(CITIES)

    rows = []
    order_id = 10001

    for day_offset in range(n_days):
        date = start_date + timedelta(days=day_offset)
        weekday = date.weekday()

        # Seasonality: more orders in Nov-Dec, fewer in summer
        month = date.month
        season_mult = {1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05,
                       6: 0.8, 7: 0.75, 8: 0.8, 9: 1.0, 10: 1.1,
                       11: 1.5, 12: 1.8}.get(month, 1.0)

        # Weekday effect: more on Mon-Wed, less on weekends
        weekday_mult = {0: 1.1, 1: 1.15, 2: 1.1, 3: 1.0,
                        4: 0.95, 5: 0.7, 6: 0.65}.get(weekday, 1.0)

        # Base orders per day
        base_orders = 130
        n_orders = int(base_orders * season_mult * weekday_mult + random.gauss(0, 10))
        n_orders = max(50, n_orders)

        for _ in range(n_orders):
            customer_id = random.randint(1, n_customers)
            city = customer_cities[customer_id]
            channel = weighted_choice(CHANNELS)

            # Category with noise
            cat_weights = {k: v["weight"] for k, v in CATEGORIES.items()}
            category = weighted_choice(cat_weights)
            cat_info = CATEGORIES[category]

            product = random.choice(cat_info["products"])
            price_min, price_max = cat_info["price_range"]
            price = round(random.uniform(price_min, price_max), -1)  # round to 10s
            quantity = random.choices([1, 2, 3, 4, 5],
                                     weights=[0.6, 0.2, 0.1, 0.05, 0.05], k=1)[0]
            revenue = price * quantity

            # Add noise: ~2% duplicate orders, ~1% missing cities, ~0.5% negative prices
            is_dup = random.random() < 0.02
            row = {
                "order_id": order_id,
                "date": date.strftime("%Y-%m-%d"),
                "customer_id": f"C{customer_id:04d}",
                "city": city if random.random() > 0.01 else "",
                "channel": channel,
                "product_category": category,
                "product": product,
                "quantity": quantity,
                "price": price if random.random() > 0.005 else -price,
                "revenue": revenue if random.random() > 0.005 else -revenue,
            }
            rows.append(row)
            if is_dup:
                rows.append(row.copy())  # duplicate
            order_id += 1

    # Shuffle slightly (keep mostly chronological)
    random.shuffle(rows)
    rows.sort(key=lambda r: r["date"])

    # Write CSV
    outpath = __import__("pathlib").Path(__file__).parent / "sales_data.csv"
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {outpath}")
    print(f"  Unique customers: {n_customers}")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Duplicates: ~{sum(1 for _ in range(100) if random.random() < 0.02)}%")

if __name__ == "__main__":
    generate()
