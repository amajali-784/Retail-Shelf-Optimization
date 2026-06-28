from __future__ import annotations

import pandas as pd

SHELF_MULTIPLIER = {
    "endcap": 1.35,
    "eye": 1.2,
    "waist": 1.08,
    "top": 0.94,
    "bottom": 0.86,
}

SHELF_ORDER = {"endcap": 1, "eye": 2, "waist": 3, "top": 4, "bottom": 5}


def optimize_shelf_layout(forecast: pd.DataFrame, products: pd.DataFrame, store_capacity_cm: int) -> pd.DataFrame:
    forecast = forecast.copy()
    products = products.copy()
    for frame in [forecast, products]:
        frame["product_id"] = frame["product_id"].astype(str)
        frame["category"] = frame["category"].astype(str)

    scored = forecast.merge(products, on=["product_id", "category"], how="left", suffixes=("", "_product"))
    scored["product_name"] = scored["product_name"].fillna("SKU " + scored["product_id"].astype(str))
    scored["category"] = scored["category"].fillna("Uncategorized").replace("", "Uncategorized")
    scored["unit_cost"] = scored["unit_cost"].fillna(scored["unit_price"] * 0.68)
    scored["margin_rate"] = scored["margin_rate"].fillna(0.32)
    scored["impulse_score"] = scored["impulse_score"].fillna(0.45)
    scored["space_required_cm"] = scored["space_required_cm"].fillna(32)
    scored["expected_margin"] = scored["predicted_units"] * (scored["unit_price"] - scored["unit_cost"])
    scored["priority_score"] = (
        scored["predicted_units"] * 0.42
        + scored["expected_margin"].clip(lower=0) * 0.26
        + scored["impulse_score"] * 80 * 0.2
        + scored["margin_rate"] * 100 * 0.12
    )
    scored = scored.sort_values("priority_score", ascending=False).copy()

    capacity = {
        "endcap": store_capacity_cm * 0.12,
        "eye": store_capacity_cm * 0.28,
        "waist": store_capacity_cm * 0.25,
        "top": store_capacity_cm * 0.18,
        "bottom": store_capacity_cm * 0.17,
    }
    used = {level: 0.0 for level in capacity}
    assignments = []
    for _, row in scored.iterrows():
        assigned = "bottom"
        for level in ["endcap", "eye", "waist", "top", "bottom"]:
            needed = float(row["space_required_cm"])
            if used[level] + needed <= capacity[level]:
                assigned = level
                used[level] += needed
                break
        assignments.append(assigned)

    scored["recommended_shelf_level"] = assignments
    scored["shelf_multiplier"] = scored["recommended_shelf_level"].map(SHELF_MULTIPLIER)
    scored["optimized_units"] = (scored["predicted_units"] * scored["shelf_multiplier"]).round(1)
    scored["incremental_units"] = (scored["optimized_units"] - scored["predicted_units"]).round(1)
    scored["incremental_revenue"] = (scored["incremental_units"] * scored["unit_price"]).round(2)
    scored["sales_lift_pct"] = ((scored["shelf_multiplier"] - 1) * 100).round(1)
    scored["placement_reason"] = scored.apply(
        lambda row: (
            f"Keep on {row['recommended_shelf_level']} shelf: forecast demand {row['predicted_units']:.1f} units, "
            f"expected sales lift {row['sales_lift_pct']:.1f}%."
            if row["sales_lift_pct"] >= 0
            else f"Keep on {row['recommended_shelf_level']} shelf because higher-value SKUs need premium space first."
        ),
        axis=1,
    )
    scored["shelf_rank"] = scored["recommended_shelf_level"].map(SHELF_ORDER).fillna(9)
    return scored.sort_values(["shelf_rank", "priority_score"], ascending=[True, False])
