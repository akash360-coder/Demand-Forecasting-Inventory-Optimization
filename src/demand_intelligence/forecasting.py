from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.demand_intelligence.feature_engineering import build_feature_matrix

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "price_index",
    "inventory_coverage",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "promotion",
    "holiday",
]


def _make_future_row(last_row: pd.Series, future_date: pd.Timestamp) -> dict[str, float | int]:
    return {
        "lag_1": float(last_row.get("lag_1", last_row["units_sold"])),
        "lag_7": float(last_row.get("lag_7", last_row["units_sold"])),
        "rolling_mean_7": float(last_row.get("rolling_mean_7", last_row["units_sold"])),
        "price_index": float(last_row.get("price_index", 1.0)),
        "inventory_coverage": float(last_row.get("inventory_coverage", 1.0)),
        "day_of_week": int(future_date.dayofweek),
        "month": int(future_date.month),
        "week_of_year": int(future_date.isocalendar().week),
        "is_weekend": int(future_date.dayofweek >= 5),
        "promotion": int(last_row.get("promotion", 0)),
        "holiday": int(last_row.get("holiday", 0)),
    }


def _fit_model(frame: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    X = frame[FEATURE_COLUMNS]
    y = frame["units_sold"]
    model.fit(X, y)
    return model


def generate_forecast_for_selection(df: pd.DataFrame, horizon: int = 14) -> list[dict[str, Any]]:
    frame = build_feature_matrix(df).copy()
    if frame.empty:
        raise ValueError("No data available to build the forecasting model.")

    model = _fit_model(frame)
    history = frame["units_sold"].tolist()
    last_generated = frame.iloc[-1].copy()
    last_date = pd.to_datetime(frame["date"].max())
    forecast_rows: list[dict[str, Any]] = []

    for offset in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=offset)
        features = _make_future_row(last_generated, future_date)
        prediction = float(model.predict(pd.DataFrame([features])[FEATURE_COLUMNS])[0])
        prediction = max(0.0, prediction)
        history.append(prediction)
        low = max(0.0, prediction - prediction * 0.2)
        high = prediction + prediction * 0.2
        forecast_rows.append(
            {
                "date": future_date,
                "historical_demand": None,
                "forecast_demand": prediction,
                "lower_bound": low,
                "upper_bound": high,
            }
        )
        last_generated = pd.Series({
            **features,
            "units_sold": prediction,
            "lag_1": prediction,
            "lag_7": history[-7] if len(history) >= 7 else prediction,
            "rolling_mean_7": float(np.mean(history[-7:])),
        })

    return forecast_rows


def compute_model_metrics(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, float]:
    frame = build_feature_matrix(df).copy()
    if len(frame) < 20:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    split_index = max(1, int(len(frame) * (1 - test_size)))
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()

    model = _fit_model(train)
    preds = model.predict(test[FEATURE_COLUMNS])
    mae = mean_absolute_error(test["units_sold"], preds)
    rmse = float(np.sqrt(mean_squared_error(test["units_sold"], preds)))
    mape = float(np.mean(np.abs((test["units_sold"] - preds) / (test["units_sold"].replace(0, np.nan).fillna(1)))) * 100)
    return {"mae": float(mae), "rmse": rmse, "mape": mape}
