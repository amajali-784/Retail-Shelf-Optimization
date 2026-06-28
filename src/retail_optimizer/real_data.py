from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RANDOM_SEED, RAW_DIR, REPORT_DIR, SHELF_LEVELS

UCI_ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download_online_retail(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "online_retail_uci.zip"
    xlsx_path = RAW_DIR / "Online Retail.xlsx"

    if force or not zip_path.exists():
        urlretrieve(UCI_ONLINE_RETAIL_URL, zip_path)

    if force or not xlsx_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(RAW_DIR)

    return xlsx_path


def _category_from_description(description: str) -> str:
    text = str(description).lower()
    rules = [
        ("Beverages", ["tea", "coffee", "mug", "cup", "bottle"]),
        ("Home Decor", ["candle", "lantern", "frame", "clock", "sign", "decoration", "heart"]),
        ("Kitchen", ["plate", "bowl", "spoon", "fork", "napkin", "cake", "kitchen", "jar"]),
        ("Stationery", ["card", "paper", "pencil", "notebook", "tag", "wrap", "sticker"]),
        ("Toys & Gifts", ["toy", "doll", "game", "set", "gift", "bag", "box"]),
        ("Apparel", ["shirt", "scarf", "hat", "sock", "glove", "jewellery", "necklace"]),
        ("Seasonal", ["christmas", "easter", "valentine", "halloween"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "General Merchandise"


def _weather_proxy(dates: pd.Series) -> pd.DataFrame:
    day = dates.dt.dayofyear
    rng = np.random.default_rng(RANDOM_SEED)
    temperature = 11 + 8 * np.sin((day - 95) / 365 * 2 * np.pi) + rng.normal(0, 1.7, len(dates))
    rainfall = np.maximum(0, 2.5 + 2.0 * np.sin((day + 20) / 365 * 2 * np.pi) + rng.normal(0, 1.3, len(dates)))
    return pd.DataFrame({"temperature_c": temperature.round(1), "rainfall_mm": rainfall.round(1)})


def _clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    df = df.rename(
        columns={
            "invoiceno": "invoice_no",
            "stockcode": "stock_code",
            "invoicedate": "invoice_date",
            "unitprice": "unit_price",
            "customerid": "customer_id",
        }
    )
    df = df.dropna(subset=["invoice_no", "stock_code", "description", "invoice_date"])
    df["invoice_no"] = df["invoice_no"].astype(str)
    df = df[~df["invoice_no"].str.startswith("C")]
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    service_pattern = "postage|manual|bank charge|discount|carriage|amazon fee|samples|adjust|dotcom"
    df = df[~df["description"].astype(str).str.lower().str.contains(service_pattern, na=False)]
    df = df[~df["stock_code"].astype(str).str.upper().isin(["POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "S"])]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["date"] = df["invoice_date"].dt.normalize()
    df["product_id"] = df["stock_code"].astype(str)
    df["product_name"] = df["description"].astype(str).str.strip().str.title()
    df["category"] = df["description"].map(_category_from_description)
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["store_id"] = df["country"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_").str.upper()
    df.loc[df["store_id"].eq("UNITED_KINGDOM"), "store_id"] = "UK_ONLINE"
    return df


def _build_products(clean: pd.DataFrame) -> pd.DataFrame:
    products = (
        clean.groupby("product_id", as_index=False)
        .agg(
            product_name=("product_name", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
            category=("category", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
            unit_price=("unit_price", "median"),
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    products = products.head(180).copy()
    margin_rate = np.select(
        [
            products["category"].eq("Home Decor"),
            products["category"].eq("Stationery"),
            products["category"].eq("Kitchen"),
            products["category"].eq("Seasonal"),
        ],
        [0.44, 0.38, 0.34, 0.42],
        default=0.3,
    )
    products["margin_rate"] = margin_rate
    products["unit_cost"] = (products["unit_price"] * (1 - products["margin_rate"])).round(2)
    products["base_daily_demand"] = (products["quantity"] / clean["date"].nunique()).round(2)
    products["impulse_score"] = (products["revenue"].rank(pct=True) * 0.65 + products["quantity"].rank(pct=True) * 0.35).round(3)
    products["space_required_cm"] = np.select(
        [products["category"].isin(["Home Decor", "Kitchen"]), products["category"].eq("Stationery")],
        [42, 24],
        default=32,
    )
    products["perishable"] = False
    return products[
        [
            "product_id",
            "product_name",
            "category",
            "unit_price",
            "unit_cost",
            "margin_rate",
            "base_daily_demand",
            "impulse_score",
            "space_required_cm",
            "perishable",
        ]
    ]


def _build_stores(clean: pd.DataFrame) -> pd.DataFrame:
    stores = (
        clean.groupby(["store_id", "country"], as_index=False)
        .agg(revenue=("revenue", "sum"), invoices=("invoice_no", "nunique"))
        .sort_values("revenue", ascending=False)
        .head(12)
    )
    stores["store_name"] = stores["country"].astype(str) + " Online Store"
    stores["region"] = stores["country"]
    stores["store_format"] = "Online Retail"
    stores["footfall_index"] = (stores["invoices"] / stores["invoices"].median()).clip(0.5, 2.2).round(3)
    stores["shelf_capacity_cm"] = 1450
    return stores[["store_id", "store_name", "region", "store_format", "footfall_index", "shelf_capacity_cm"]]


def _build_daily_sales(clean: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    product_ids = set(products["product_id"])
    store_ids = set(stores["store_id"])
    df = clean[clean["product_id"].isin(product_ids) & clean["store_id"].isin(store_ids)].copy()
    daily = (
        df.groupby(["date", "store_id", "product_id", "category"], as_index=False)
        .agg(units_sold=("quantity", "sum"), revenue=("revenue", "sum"), unit_price=("unit_price", "median"))
    )
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    product_store = daily[["store_id", "product_id", "category"]].drop_duplicates()
    full_index = product_store.merge(pd.DataFrame({"date": all_dates}), how="cross")
    daily = full_index.merge(daily, on=["date", "store_id", "product_id", "category"], how="left")
    daily["units_sold"] = daily["units_sold"].fillna(0).astype(int)
    daily = daily.merge(products[["product_id", "unit_price", "unit_cost"]], on="product_id", how="left", suffixes=("", "_product"))
    daily["unit_price"] = daily["unit_price"].fillna(daily["unit_price_product"])
    daily["revenue"] = daily["revenue"].fillna(daily["units_sold"] * daily["unit_price"])
    daily["gross_profit"] = daily["units_sold"] * (daily["unit_price"] - daily["unit_cost"])
    daily["discount_pct"] = np.where(daily["date"].dt.month.isin([11, 12]), rng.choice([0, 5, 10, 15], len(daily), p=[0.72, 0.12, 0.1, 0.06]), 0)
    daily["promotion_flag"] = (daily["discount_pct"] > 0).astype(int)
    daily["shelf_level"] = rng.choice(SHELF_LEVELS, len(daily), p=[0.18, 0.22, 0.28, 0.2, 0.12])
    daily["stock_on_hand"] = (daily["units_sold"] * rng.uniform(1.4, 2.8, len(daily)) + rng.integers(4, 30, len(daily))).astype(int)
    daily["is_weekend"] = (daily["date"].dt.dayofweek >= 5).astype(int)
    daily["is_holiday"] = daily["date"].dt.strftime("%m-%d").isin(["01-01", "12-24", "12-25", "12-26", "12-31"]).astype(int)
    daily = daily.sort_values(["store_id", "product_id", "date"])
    grouped_units = daily.groupby(["store_id", "product_id"])["units_sold"]
    daily["lag_7_units"] = grouped_units.shift(7).fillna(0)
    daily["rolling_7_units"] = grouped_units.transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean()).fillna(0).round(2)
    daily["rolling_28_units"] = grouped_units.transform(lambda values: values.shift(1).rolling(28, min_periods=1).mean()).fillna(0).round(2)
    weather = _weather_proxy(daily["date"])
    daily["temperature_c"] = weather["temperature_c"]
    daily["rainfall_mm"] = weather["rainfall_mm"]
    daily["date"] = daily["date"].dt.date.astype(str)
    return daily[
        [
            "date",
            "store_id",
            "product_id",
            "category",
            "unit_price",
            "discount_pct",
            "promotion_flag",
            "shelf_level",
            "temperature_c",
            "rainfall_mm",
            "is_weekend",
            "is_holiday",
            "stock_on_hand",
            "lag_7_units",
            "rolling_7_units",
            "rolling_28_units",
            "units_sold",
            "revenue",
            "gross_profit",
        ]
    ]


def _build_customer_baskets(clean: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    df = clean[clean["product_id"].isin(products["product_id"]) & clean["store_id"].isin(stores["store_id"])].copy()
    df["basket_id"] = df["invoice_no"]
    return df[["basket_id", "date", "store_id", "product_id", "product_name", "category", "quantity"]]


def import_online_retail_dataset(force_download: bool = False) -> dict[str, Path]:
    xlsx_path = download_online_retail(force=force_download)
    raw = pd.read_excel(xlsx_path)
    clean = _clean_transactions(raw)
    products = _build_products(clean)
    stores = _build_stores(clean)
    daily = _build_daily_sales(clean, products, stores)
    baskets = _build_customer_baskets(clean, products, stores)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    clean.to_csv(PROCESSED_DIR / "clean_online_retail_transactions.csv", index=False)
    products.to_csv(RAW_DIR / "products.csv", index=False)
    stores.to_csv(RAW_DIR / "stores.csv", index=False)
    daily.to_csv(PROCESSED_DIR / "retail_daily.csv", index=False)
    baskets.to_csv(PROCESSED_DIR / "customer_baskets.csv", index=False)

    profile = {
        "source": "UCI Machine Learning Repository - Online Retail",
        "source_url": UCI_ONLINE_RETAIL_URL,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "products_selected": int(len(products)),
        "stores_selected": int(len(stores)),
        "daily_rows": int(len(daily)),
        "basket_rows": int(len(baskets)),
        "date_min": str(clean["date"].min().date()),
        "date_max": str(clean["date"].max().date()),
    }
    with open(REPORT_DIR / "dataset_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return {
        "transactions": PROCESSED_DIR / "clean_online_retail_transactions.csv",
        "products": RAW_DIR / "products.csv",
        "stores": RAW_DIR / "stores.csv",
        "retail_daily": PROCESSED_DIR / "retail_daily.csv",
        "customer_baskets": PROCESSED_DIR / "customer_baskets.csv",
        "profile": REPORT_DIR / "dataset_profile.json",
    }
