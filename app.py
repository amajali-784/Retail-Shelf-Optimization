from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retail_optimizer.association_rules import mine_association_rules
from retail_optimizer.config import MODEL_DIR, PROCESSED_DIR, RAW_DIR, REPORT_DIR
from retail_optimizer.modeling import forecast_next_week
from retail_optimizer.recommendations import recommend_discounts
from retail_optimizer.shelf_optimizer import optimize_shelf_layout

st.set_page_config(page_title="Retail Shelf Optimization", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(PROCESSED_DIR / "retail_daily.csv", parse_dates=["date"])
    products = pd.read_csv(RAW_DIR / "products.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    baskets = pd.read_csv(PROCESSED_DIR / "customer_baskets.csv")
    for frame in [sales, products, baskets]:
        frame["product_id"] = frame["product_id"].astype(str)
    sales["store_id"] = sales["store_id"].astype(str)
    stores["store_id"] = stores["store_id"].astype(str)
    baskets["store_id"] = baskets["store_id"].astype(str)
    products["product_name"] = products["product_name"].fillna("SKU " + products["product_id"])
    return sales, products, stores, baskets


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "sales_model.joblib")


def require_artifacts() -> None:
    required = [
        PROCESSED_DIR / "retail_daily.csv",
        PROCESSED_DIR / "customer_baskets.csv",
        RAW_DIR / "products.csv",
        RAW_DIR / "stores.csv",
        MODEL_DIR / "sales_model.joblib",
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        st.error("Missing project artifacts. Run `python scripts/generate_dataset.py` and `python scripts/train_models.py`.")
        st.code("\n".join(missing), language="text")
        st.stop()


require_artifacts()
sales, products, stores, baskets = load_data()
model = load_model()

st.title("Retail Shelf Optimization ML Suite")
st.caption("Demand forecasting, sales prediction, shelf placement, weather effects, discount strategy, and basket insights.")

INDIA_STORE_ID = "INDIA_MARKET"
INDIA_CATEGORY_LIFT = {
    "Beverages": 1.25,
    "Kitchen": 1.18,
    "Stationery": 1.2,
    "Toys & Gifts": 1.16,
    "Seasonal": 1.22,
    "Home Decor": 1.12,
    "Apparel": 1.08,
    "General Merchandise": 1.1,
}


def build_india_market(
    base_sales: pd.DataFrame, base_stores: pd.DataFrame, base_baskets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    india_sales = base_sales[base_sales["store_id"].eq("UK_ONLINE")].copy()
    if india_sales.empty:
        india_sales = base_sales.copy()
    india_sales["store_id"] = INDIA_STORE_ID
    india_sales["temperature_c"] = india_sales["temperature_c"] + 9
    india_sales["rainfall_mm"] = india_sales["rainfall_mm"] * 1.35
    india_sales["unit_price"] = india_sales["unit_price"] * 83
    india_sales["revenue"] = india_sales["revenue"] * 83
    india_sales["gross_profit"] = india_sales["gross_profit"] * 83
    lift = india_sales["category"].map(INDIA_CATEGORY_LIFT).fillna(1.1)
    india_sales["units_sold"] = (india_sales["units_sold"] * lift).round().astype(int)
    india_sales["stock_on_hand"] = (india_sales["stock_on_hand"] * lift).round().astype(int)
    india_store = pd.DataFrame(
        [
            {
                "store_id": INDIA_STORE_ID,
                "store_name": "India Retail Scenario",
                "region": "India",
                "store_format": "Projected Omnichannel",
                "footfall_index": 1.65,
                "shelf_capacity_cm": 1650,
            }
        ]
    )
    india_baskets = base_baskets[base_baskets["store_id"].eq("UK_ONLINE")].copy()
    if india_baskets.empty:
        india_baskets = base_baskets.copy()
    india_baskets["store_id"] = INDIA_STORE_ID
    india_baskets["basket_id"] = "IN_" + india_baskets["basket_id"].astype(str)
    return (
        pd.concat([base_sales, india_sales], ignore_index=True),
        pd.concat([base_stores, india_store], ignore_index=True),
        pd.concat([base_baskets, india_baskets], ignore_index=True),
    )


sales, stores, baskets = build_india_market(sales, stores, baskets)
store_options = stores["store_id"].tolist()
category_options = ["All"] + sorted(products["category"].unique().tolist())
product_options = ["All"] + products.sort_values("product_name")["product_name"].tolist()

with st.sidebar:
    st.header("Controls")
    selected_store = st.selectbox("Store", store_options)
    selected_category = st.selectbox("Category", category_options)
    selected_product_name = st.selectbox("Product", product_options)
    lookback_days = st.slider("Lookback days", 30, 365, 180, step=15)

filtered = sales[sales["store_id"] == selected_store].copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if selected_product_name != "All":
    selected_product_id = products.loc[products["product_name"].eq(selected_product_name), "product_id"].iloc[0]
    filtered = filtered[filtered["product_id"].eq(selected_product_id)]
filtered = filtered[filtered["date"] >= filtered["date"].max() - pd.Timedelta(days=lookback_days)]

forecast = forecast_next_week(model, sales[sales["store_id"] == selected_store])
if selected_store == INDIA_STORE_ID:
    india_lift = forecast["category"].map(INDIA_CATEGORY_LIFT).fillna(1.1)
    forecast["predicted_units"] = (forecast["predicted_units"] * india_lift).round(1)
    forecast["predicted_revenue"] = (forecast["predicted_units"] * forecast["unit_price"]).round(2)
if selected_category != "All":
    forecast = forecast[forecast["category"] == selected_category]
if selected_product_name != "All":
    forecast = forecast[forecast["product_id"].eq(selected_product_id)]

store_capacity = int(stores.loc[stores["store_id"] == selected_store, "shelf_capacity_cm"].iloc[0])
shelf_plan = optimize_shelf_layout(forecast, products, store_capacity)
discounts = recommend_discounts(forecast, products)

sales_total = filtered["revenue"].sum()
units_total = filtered["units_sold"].sum()
profit_total = filtered["gross_profit"].sum()
forecast_total = forecast["predicted_revenue"].sum()
currency_symbol = "₹" if selected_store == INDIA_STORE_ID else "$"

metric_cols = st.columns(4)
metric_cols[0].metric("Revenue", f"{currency_symbol}{sales_total:,.0f}")
metric_cols[1].metric("Units Sold", f"{units_total:,.0f}")
metric_cols[2].metric("Gross Profit", f"{currency_symbol}{profit_total:,.0f}")
metric_cols[3].metric("Next Week Forecast", f"{currency_symbol}{forecast_total:,.0f}")

tab_overview, tab_eda, tab_forecast, tab_shelf, tab_discounts, tab_patterns, tab_model = st.tabs(
    ["Overview", "EDA", "Forecast", "Shelf Plan", "Discounts", "Purchase Patterns", "Model"]
)

with tab_overview:
    daily = filtered.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
    category = filtered.groupby("category", as_index=False).agg(revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"))
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(px.line(daily, x="date", y="revenue", title="Daily Revenue"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(category, x="revenue", y="category", orientation="h", title="Revenue by Category"), use_container_width=True)

    weather = filtered.groupby("temperature_c", as_index=False).agg(units_sold=("units_sold", "mean"))
    st.plotly_chart(px.scatter(weather, x="temperature_c", y="units_sold", title="Weather Demand Signal"), use_container_width=True)

with tab_eda:
    profile_path = REPORT_DIR / "dataset_profile.json"
    summary_path = REPORT_DIR / "eda_summary.json"
    monthly_path = REPORT_DIR / "eda_monthly_revenue.csv"
    country_path = REPORT_DIR / "eda_country_revenue.csv"
    product_path = REPORT_DIR / "eda_top_products.csv"

    left, right = st.columns(2)
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as file:
            left.json(json.load(file))
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            right.json(json.load(file))

    if monthly_path.exists():
        monthly = pd.read_csv(monthly_path)
        st.plotly_chart(px.line(monthly, x="month", y="revenue", title="EDA: Monthly Revenue"), use_container_width=True)
    if country_path.exists():
        country = pd.read_csv(country_path).head(12)
        st.plotly_chart(px.bar(country, x="revenue", y="country", orientation="h", title="EDA: Top Countries"), use_container_width=True)
    if product_path.exists():
        top_products = pd.read_csv(product_path).head(20)
        st.dataframe(top_products, use_container_width=True, hide_index=True)

with tab_forecast:
    forecast_view = forecast.merge(products[["product_id", "product_name", "margin_rate"]], on="product_id", how="left")
    top = forecast_view.sort_values("predicted_revenue", ascending=False).head(20)
    st.plotly_chart(
        px.bar(top, x="predicted_revenue", y="product_name", color="category", orientation="h", title="Top Forecasted Products"),
        use_container_width=True,
    )
    st.dataframe(
        forecast_view[
            ["store_id", "product_name", "category", "predicted_units", "predicted_revenue", "unit_price", "margin_rate"]
        ].sort_values("predicted_revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with tab_shelf:
    plan_cols = [
        "product_name",
        "category",
        "recommended_shelf_level",
        "predicted_units",
        "optimized_units",
        "incremental_units",
        "incremental_revenue",
        "sales_lift_pct",
        "placement_reason",
        "priority_score",
    ]
    if shelf_plan.empty:
        st.info("No products match the selected filters.")
    else:
        recommendation = shelf_plan.sort_values("incremental_revenue", ascending=False).iloc[0]
        st.subheader("Best Placement Recommendation")
        st.write(
            f"**{recommendation['product_name']}** should be kept on the "
            f"**{recommendation['recommended_shelf_level']} shelf**. "
            f"Expected sales lift is **{recommendation['sales_lift_pct']:.1f}%**, "
            f"adding about **{recommendation['incremental_units']:.1f} units** and "
            f"**{currency_symbol}{recommendation['incremental_revenue']:,.0f}** in revenue."
        )

        chart_plan = shelf_plan.copy()
        chart_plan["product_label"] = (
            chart_plan["product_name"].fillna("SKU " + chart_plan["product_id"].astype(str)).astype(str).str.strip()
            + " [" + chart_plan["product_id"].astype(str) + "]"
        )
        chart_plan["category"] = chart_plan["category"].fillna("Uncategorized").replace("", "Uncategorized")
        chart_plan = chart_plan[chart_plan["optimized_units"].gt(0)]
        st.plotly_chart(
            px.treemap(
                chart_plan,
                path=["recommended_shelf_level", "category", "product_label"],
                values="optimized_units",
                color="incremental_revenue",
                title="Optimized Shelf Allocation",
            ),
            use_container_width=True,
        )
        display_plan = shelf_plan[plan_cols].copy()
        numeric_cols = display_plan.select_dtypes("number").columns
        display_plan[numeric_cols] = display_plan[numeric_cols].round(2)
        st.dataframe(display_plan, use_container_width=True, hide_index=True)

with tab_discounts:
    products["product_id"] = products["product_id"].astype(str)
    discounts["product_id"] = discounts["product_id"].astype(str)
    discount_view = discounts.merge(products[["product_id", "product_name", "category"]], on="product_id", how="left", suffixes=("", "_product"))
    suggested = discount_view[discount_view["recommended_discount"] > 0].head(20)
    st.plotly_chart(
        px.bar(
            suggested,
            x="expected_profit_after_discount",
            y="product_name",
            color="recommended_discount",
            orientation="h",
            title="Recommended Targeted Discounts",
        ),
        use_container_width=True,
    )
    st.dataframe(
        discount_view[
            ["product_name", "category", "predicted_units", "recommended_discount", "discount_action", "expected_profit_after_discount"]
        ].sort_values("expected_profit_after_discount", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with tab_patterns:
    rules = mine_association_rules(baskets[baskets["store_id"] == selected_store])
    st.dataframe(rules.head(30), use_container_width=True, hide_index=True)
    if not rules.empty:
        st.plotly_chart(
            px.scatter(rules.head(30), x="support", y="confidence", size="lift", color="antecedent", hover_data=["consequent"]),
            use_container_width=True,
        )
    else:
        st.info("No strong product-pair rules found for the selected filters.")

with tab_model:
    metrics_path = REPORT_DIR / "model_metrics.json"
    importance_path = REPORT_DIR / "feature_importance.csv"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        cols = st.columns(3)
        cols[0].metric("MAE", metrics["mae_units"])
        cols[1].metric("WAPE", metrics.get("wape", "n/a"))
        cols[2].metric("Baseline MAE", metrics.get("baseline_mae_units", "n/a"))
        st.json(metrics)
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(20)
        st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance"), use_container_width=True)
