from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from .features import feature_columns, prepare_sales_features


def build_model() -> Pipeline:
    categorical = ["store_id", "product_id", "category", "shelf_level"]
    numeric = [col for col in feature_columns() if col not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ]
    )
    model = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.07, l2_regularization=0.03, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_importance_model() -> Pipeline:
    categorical = ["store_id", "product_id", "category", "shelf_level"]
    numeric = [col for col in feature_columns() if col not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ],
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(n_estimators=120, min_samples_leaf=12, random_state=42, n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_sales_model(data_path: Path = PROCESSED_DIR / "retail_daily.csv") -> dict[str, float]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_sales_features(pd.read_csv(data_path, low_memory=False))
    cutoff = df["date"].max() - pd.Timedelta(days=56)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    x_train = train[feature_columns()]
    y_train = train["units_sold"]
    x_test = test[feature_columns()]
    y_test = test["units_sold"]

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).clip(min=0)
    baseline = x_test["rolling_28_units"].fillna(y_train.mean()).to_numpy()

    metrics = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "mae_units": round(float(mean_absolute_error(y_test, predictions)), 3),
        "baseline_mae_units": round(float(mean_absolute_error(y_test, baseline)), 3),
        "rmse_units": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
        "wape": round(float(np.sum(np.abs(y_test - predictions)) / max(np.sum(np.abs(y_test)), 1)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "validation_start": cutoff.date().isoformat(),
        "validation_end": df["date"].max().date().isoformat(),
        "note": "SKU-level retail demand is sparse and intermittent; use MAE/WAPE alongside R2.",
    }

    joblib.dump(model, MODEL_DIR / "sales_model.joblib")

    importance_model = build_importance_model()
    sample = train.sample(min(25000, len(train)), random_state=42)
    importance_model.fit(sample[feature_columns()], sample["units_sold"])
    encoded_names = importance_model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": encoded_names,
            "importance": importance_model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.head(30).to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    with open(REPORT_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def forecast_next_week(model: Pipeline, sales: pd.DataFrame) -> pd.DataFrame:
    df = prepare_sales_features(sales)
    latest = df.sort_values("date").groupby(["store_id", "product_id"], as_index=False).tail(1)
    latest = latest.copy()
    latest["date"] = latest["date"] + pd.Timedelta(days=7)
    latest["discount_pct"] = 0
    latest["promotion_flag"] = 0
    latest = prepare_sales_features(latest)
    latest["predicted_units"] = model.predict(latest[feature_columns()]).clip(min=0).round(1)
    latest["predicted_revenue"] = (latest["predicted_units"] * latest["unit_price"]).round(2)
    return latest
