from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, REPORT_DIR


def run_eda(transactions_path: Path = PROCESSED_DIR / "clean_online_retail_transactions.csv") -> dict[str, object]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(transactions_path, parse_dates=["invoice_date", "date"], low_memory=False)

    monthly = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), invoices=("invoice_no", "nunique"))
    )
    country = df.groupby("country", as_index=False).agg(revenue=("revenue", "sum"), invoices=("invoice_no", "nunique")).sort_values("revenue", ascending=False)
    products = (
        df.groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), invoices=("invoice_no", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    customers = (
        df.dropna(subset=["customer_id"])
        .groupby("customer_id", as_index=False)
        .agg(revenue=("revenue", "sum"), invoices=("invoice_no", "nunique"), quantity=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
    )

    monthly.to_csv(REPORT_DIR / "eda_monthly_revenue.csv", index=False)
    country.to_csv(REPORT_DIR / "eda_country_revenue.csv", index=False)
    products.head(50).to_csv(REPORT_DIR / "eda_top_products.csv", index=False)
    customers.head(50).to_csv(REPORT_DIR / "eda_top_customers.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "revenue": round(float(df["revenue"].sum()), 2),
        "quantity": int(df["quantity"].sum()),
        "invoices": int(df["invoice_no"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "countries": int(df["country"].nunique()),
        "products": int(df["product_id"].nunique()),
        "avg_order_value": round(float(df.groupby("invoice_no")["revenue"].sum().mean()), 2),
        "top_country": str(country.iloc[0]["country"]),
        "top_product": str(products.iloc[0]["product_name"]),
    }
    with open(REPORT_DIR / "eda_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
