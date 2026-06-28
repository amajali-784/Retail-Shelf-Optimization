from __future__ import annotations

import pandas as pd


def recommend_discounts(forecast: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = forecast.merge(products[["product_id", "unit_cost", "margin_rate", "perishable"]], on="product_id", how="left")
    candidates = []
    for _, row in df.iterrows():
        best = {"recommended_discount": 0, "expected_profit": row["predicted_units"] * (row["unit_price"] - row["unit_cost"])}
        for discount in [0, 5, 10, 15, 20, 25]:
            elasticity = 0.9 + row["margin_rate"] * 1.6 + (0.35 if row["perishable"] else 0)
            uplift_units = row["predicted_units"] * (1 + discount / 100 * elasticity)
            discounted_price = row["unit_price"] * (1 - discount / 100)
            profit = uplift_units * (discounted_price - row["unit_cost"])
            if profit > best["expected_profit"]:
                best = {"recommended_discount": discount, "expected_profit": profit}
        candidates.append(best)
    out = df.copy()
    out["recommended_discount"] = [item["recommended_discount"] for item in candidates]
    out["expected_profit_after_discount"] = [round(item["expected_profit"], 2) for item in candidates]
    out["discount_action"] = out["recommended_discount"].map(
        lambda value: "No discount" if value == 0 else f"{value}% targeted discount"
    )
    return out.sort_values("expected_profit_after_discount", ascending=False)
