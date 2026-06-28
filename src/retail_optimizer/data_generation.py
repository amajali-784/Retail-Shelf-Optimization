from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORIES, PROCESSED_DIR, RANDOM_SEED, RAW_DIR, SHELF_LEVELS


@dataclass(frozen=True)
class DatasetSpec:
    stores: int = 5
    products_per_category: int = 6
    days: int = 730
    baskets: int = 18000


def _season_multiplier(date: pd.Timestamp, category: str) -> float:
    month = date.month
    if category == "Beverages":
        return 1.25 if month in [4, 5, 6, 7, 8] else 0.95
    if category == "Frozen":
        return 1.18 if month in [5, 6, 7, 8] else 0.96
    if category == "Bakery":
        return 1.16 if month in [11, 12] else 1.0
    if category == "Fresh Produce":
        return 1.12 if month in [1, 2, 3, 9, 10] else 0.98
    if category == "Household":
        return 1.12 if month in [3, 4, 10, 11] else 1.0
    return 1.0


def build_products(rng: np.random.Generator, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    product_id = 1000
    for category in CATEGORIES:
        for idx in range(1, spec.products_per_category + 1):
            price = rng.uniform(2.0, 18.0)
            if category in ["Household", "Personal Care"]:
                price *= rng.uniform(1.5, 2.6)
            margin_rate = rng.uniform(0.18, 0.48)
            rows.append(
                {
                    "product_id": f"P{product_id}",
                    "product_name": f"{category} Item {idx}",
                    "category": category,
                    "unit_price": round(price, 2),
                    "unit_cost": round(price * (1 - margin_rate), 2),
                    "margin_rate": round(margin_rate, 3),
                    "base_daily_demand": round(rng.uniform(12, 80), 1),
                    "impulse_score": round(rng.uniform(0.1, 0.95), 3),
                    "space_required_cm": int(rng.integers(18, 55)),
                    "perishable": category in ["Fresh Produce", "Dairy", "Bakery"],
                }
            )
            product_id += 1
    return pd.DataFrame(rows)


def build_stores(rng: np.random.Generator, spec: DatasetSpec) -> pd.DataFrame:
    formats = ["Urban Express", "Suburban Market", "Flagship", "Neighborhood"]
    rows = []
    for store_idx in range(1, spec.stores + 1):
        rows.append(
            {
                "store_id": f"S{store_idx:02d}",
                "store_name": f"Store {store_idx:02d}",
                "region": rng.choice(["North", "South", "East", "West"]),
                "store_format": rng.choice(formats),
                "footfall_index": round(rng.uniform(0.75, 1.45), 3),
                "shelf_capacity_cm": int(rng.integers(1100, 1700)),
            }
        )
    return pd.DataFrame(rows)


def build_daily_sales(
    rng: np.random.Generator, products: pd.DataFrame, stores: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=spec.days)
    rows = []
    shelf_effect = {"bottom": 0.82, "waist": 1.03, "eye": 1.16, "top": 0.9, "endcap": 1.32}
    for date in dates:
        day_of_week = date.dayofweek
        weekend = day_of_week >= 5
        holiday = (date.month, date.day) in [(1, 1), (7, 4), (11, 27), (12, 24), (12, 25), (12, 31)]
        for _, store in stores.iterrows():
            temp = 22 + 10 * np.sin((date.dayofyear - 80) / 365 * 2 * np.pi) + rng.normal(0, 4)
            rainfall = max(0, rng.gamma(1.2, 3.0) - 2.5)
            for _, product in products.iterrows():
                promo = rng.random() < (0.16 if product["category"] in ["Snacks", "Beverages"] else 0.1)
                discount_pct = rng.choice([0, 5, 10, 15, 20, 25], p=[0.68, 0.08, 0.1, 0.08, 0.04, 0.02])
                if not promo:
                    discount_pct = 0
                shelf_level = rng.choice(SHELF_LEVELS, p=[0.18, 0.22, 0.28, 0.2, 0.12])
                seasonal = _season_multiplier(date, product["category"])
                weather = 1.0
                if product["category"] in ["Beverages", "Frozen"]:
                    weather += max(temp - 24, 0) * 0.018
                if product["category"] in ["Bakery", "Dairy"]:
                    weather -= max(temp - 29, 0) * 0.012
                if product["category"] in ["Household", "Personal Care"]:
                    weather += min(rainfall, 15) * 0.008
                demand = (
                    product["base_daily_demand"]
                    * store["footfall_index"]
                    * seasonal
                    * weather
                    * (1.13 if weekend else 1.0)
                    * (1.22 if holiday else 1.0)
                    * shelf_effect[shelf_level]
                    * (1 + discount_pct / 100 * rng.uniform(0.8, 1.7))
                )
                units = max(0, int(rng.normal(demand, max(2.5, demand * 0.18))))
                stock_on_hand = max(units, int(rng.normal(demand * rng.uniform(1.3, 2.6), 12)))
                unit_price = product["unit_price"] * (1 - discount_pct / 100)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": store["store_id"],
                        "product_id": product["product_id"],
                        "category": product["category"],
                        "unit_price": round(unit_price, 2),
                        "discount_pct": discount_pct,
                        "promotion_flag": int(discount_pct > 0),
                        "shelf_level": shelf_level,
                        "temperature_c": round(temp, 1),
                        "rainfall_mm": round(rainfall, 1),
                        "is_weekend": int(weekend),
                        "is_holiday": int(holiday),
                        "stock_on_hand": stock_on_hand,
                        "units_sold": units,
                        "revenue": round(units * unit_price, 2),
                        "gross_profit": round(units * (unit_price - product["unit_cost"]), 2),
                    }
                )
    return pd.DataFrame(rows)


def build_customer_baskets(
    rng: np.random.Generator, products: pd.DataFrame, stores: pd.DataFrame, sales: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    product_lookup = products.set_index("product_id")
    category_pairs = {
        "Fresh Produce": ["Dairy", "Bakery"],
        "Dairy": ["Bakery", "Fresh Produce"],
        "Bakery": ["Dairy", "Beverages"],
        "Beverages": ["Snacks", "Frozen"],
        "Snacks": ["Beverages", "Personal Care"],
        "Household": ["Personal Care"],
        "Personal Care": ["Household"],
        "Frozen": ["Beverages", "Snacks"],
    }
    dates = pd.to_datetime(sales["date"]).drop_duplicates().to_numpy()
    rows = []
    for basket_idx in range(spec.baskets):
        basket_id = f"B{basket_idx + 1:06d}"
        store_id = rng.choice(stores["store_id"])
        basket_date = pd.Timestamp(rng.choice(dates)).date().isoformat()
        first = products.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        basket_size = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.18, 0.3, 0.24, 0.16, 0.08, 0.04]))
        selected = [first["product_id"]]
        for _ in range(basket_size - 1):
            related_categories = category_pairs[first["category"]]
            pool = products[products["category"].isin(related_categories)]
            if rng.random() < 0.22:
                pool = products
            selected.append(rng.choice(pool["product_id"]))
        for product_id in sorted(set(selected)):
            rows.append(
                {
                    "basket_id": basket_id,
                    "date": basket_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "product_name": product_lookup.loc[product_id, "product_name"],
                    "category": product_lookup.loc[product_id, "category"],
                    "quantity": int(rng.choice([1, 1, 1, 2, 3], p=[0.56, 0.18, 0.1, 0.12, 0.04])),
                }
            )
    return pd.DataFrame(rows)


def generate_dataset(output_dir: Path = PROCESSED_DIR, spec: DatasetSpec | None = None) -> dict[str, Path]:
    spec = spec or DatasetSpec()
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    products = build_products(rng, spec)
    stores = build_stores(rng, spec)
    sales = build_daily_sales(rng, products, stores, spec)
    baskets = build_customer_baskets(rng, products, stores, sales, spec)

    paths = {
        "products": RAW_DIR / "products.csv",
        "stores": RAW_DIR / "stores.csv",
        "retail_daily": output_dir / "retail_daily.csv",
        "customer_baskets": output_dir / "customer_baskets.csv",
    }
    products.to_csv(paths["products"], index=False)
    stores.to_csv(paths["stores"], index=False)
    sales.to_csv(paths["retail_daily"], index=False)
    baskets.to_csv(paths["customer_baskets"], index=False)
    return paths
