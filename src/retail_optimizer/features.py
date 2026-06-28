from __future__ import annotations

import pandas as pd

SHELF_SCORE = {"bottom": 1, "top": 2, "waist": 3, "eye": 4, "endcap": 5}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    return out


def prepare_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    for column in ["store_id", "product_id", "category", "shelf_level"]:
        out[column] = out[column].astype(str)
    out["shelf_score"] = out["shelf_level"].map(SHELF_SCORE).fillna(2)
    out["inventory_pressure"] = out["stock_on_hand"] / (out["units_sold"].clip(lower=1))
    for column, default in [("lag_7_units", 0), ("rolling_7_units", 0), ("rolling_28_units", 0)]:
        if column not in out.columns:
            out[column] = default
    return out


def feature_columns() -> list[str]:
    return [
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
        "day_of_week",
        "week_of_year",
        "month",
        "quarter",
        "shelf_score",
        "inventory_pressure",
        "lag_7_units",
        "rolling_7_units",
        "rolling_28_units",
    ]
