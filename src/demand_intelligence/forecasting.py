from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.demand_intelligence.evaluation import evaluate_predictions
from src.demand_intelligence.feature_engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    detect_seasonal_period,
    validate_feature_columns,
)
from src.demand_intelligence.leakage_detection import validate_no_leakage

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_MODEL_PATH = MODEL_DIR / "production_forecast_model.joblib"


def _make_regressor(model_name: str):
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed.")
        return XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed.")
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_model(frame: pd.DataFrame, model_name: str = "Random Forest") -> Any:
    validate_no_leakage(frame)
    model = _make_regressor(model_name)
    X = frame[FEATURE_COLUMNS]
    y = frame["units_sold"]
    model.fit(X, y)
    return model


def naive_forecast(history: pd.Series, horizon: int) -> pd.Series:
    last_value = float(history.iloc[-1]) if len(history) else 0.0
    return pd.Series([max(0.0, last_value) for _ in range(horizon)], dtype=float)


def seasonal_naive_forecast(history: pd.Series, horizon: int, period: int | None = None) -> pd.Series:
    history = history.astype(float)
    seasonal_period = period or detect_seasonal_period(pd.DataFrame({"date": pd.date_range("2000-01-01", periods=len(history), freq="D"), "units_sold": history}))
    if seasonal_period <= 0:
        seasonal_period = 1
    pattern = history.iloc[-seasonal_period:].to_numpy()
    values = []
    for step in range(horizon):
        index = step % seasonal_period
        values.append(max(0.0, float(pattern[index])))
    return pd.Series(values, dtype=float)


def moving_average_forecast(history: pd.Series, horizon: int, window: int | None = None) -> pd.Series:
    history = history.astype(float)
    window_size = window or min(7, max(3, len(history)))
    last_window = history.iloc[-window_size:]
    average = float(last_window.mean()) if len(last_window) else 0.0
    return pd.Series([max(0.0, average) for _ in range(horizon)], dtype=float)


def time_series_split(df: pd.DataFrame, validation_days: int = 30, test_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        raise ValueError("Time-series split requires a date column.")
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Time-series split cannot contain missing dates.")
    validate_feature_columns(FEATURE_COLUMNS)
    frame = frame.sort_values("date", kind="mergesort").copy()
    if len(frame) < 45:
        raise ValueError("At least 45 days of data are required for time-based validation.")

    max_date = frame["date"].max()
    if len(frame) <= validation_days + test_days:
        validation_days = max(7, len(frame) // 4)
        test_days = max(7, len(frame) // 6)

    test_start = max_date - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = frame[frame["date"] < validation_start].copy()
    validation = frame[(frame["date"] >= validation_start) & (frame["date"] < test_start)].copy()
    test = frame[frame["date"] >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Unable to create non-empty chronological train/validation/test splits.")

    if not (train["date"].max() < validation["date"].min() and validation["date"].max() < test["date"].min()):
        raise ValueError("Chronological split invariant violated: partitions overlap or are out of order.")

    return train, validation, test


def _build_future_row(history_values: list[float], future_date: pd.Timestamp, last_observed: pd.Series) -> dict[str, float | int]:
    past = [max(0.0, float(x)) for x in history_values]
    return {
        "lag_1": float(past[-1]) if past else 0.0,
        "lag_7": float(past[-7]) if len(past) >= 7 else float(past[-1]) if past else 0.0,
        "lag_14": float(past[-14]) if len(past) >= 14 else float(past[-1]) if past else 0.0,
        "rolling_mean_7": float(np.mean(past[-7:])) if len(past) >= 1 else 0.0,
        "rolling_mean_14": float(np.mean(past[-14:])) if len(past) >= 1 else 0.0,
        "rolling_std_7": float(np.std(past[-7:], ddof=0)) if len(past) >= 2 else 0.0,
        "day_of_week": int(future_date.dayofweek),
        "month": int(future_date.month),
        "week_of_year": int(future_date.isocalendar().week),
        "day_of_year": int(future_date.dayofyear),
        "quarter": int(future_date.quarter),
        "is_weekend": int(future_date.dayofweek >= 5),
        "promotion": int(last_observed.get("promotion", 0)),
        "holiday": int(last_observed.get("holiday", 0)),
        "price": float(last_observed.get("price", 0.0)),
        "price_change": float(last_observed.get("price_change", 0.0)),
        "inventory_on_hand_lag_1": float(last_observed.get("inventory_on_hand_lag_1", last_observed.get("inventory_on_hand", 0.0))),
        "lead_time_days": float(last_observed.get("lead_time_days", last_observed.get("supplier_lead_time_days", 0.0))),
    }


def _baseline_predictions(history: pd.Series, horizon: int, model_name: str) -> pd.Series:
    if model_name == "Naive":
        return naive_forecast(history, horizon)
    if model_name == "Seasonal Naive":
        return seasonal_naive_forecast(history, horizon, detect_seasonal_period(pd.DataFrame({"date": pd.date_range("2000-01-01", periods=len(history), freq="D"), "units_sold": history})))
    if model_name == "Moving Average":
        return moving_average_forecast(history, horizon, window=min(7, max(3, len(history))))
    raise ValueError(f"Unsupported baseline model: {model_name}")


def _model_results_for_split(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    model = _fit_model(frame, model_name=model_name)
    validation_metrics = evaluate_predictions(frame["units_sold"].to_numpy(), model.predict(frame[FEATURE_COLUMNS]))
    return {"model": model, "metrics": validation_metrics}


def run_experiment(df: pd.DataFrame) -> dict[str, Any]:
    frame = build_feature_matrix(df).copy()
    if frame.empty:
        raise ValueError("No valid feature rows are available for time-series training.")

    train, validation, test = time_series_split(frame)
    candidates = ["Naive", "Seasonal Naive", "Moving Average", "Random Forest", "XGBoost", "LightGBM"]
    results: list[dict[str, Any]] = []

    for model_name in candidates:
        if model_name in {"Naive", "Seasonal Naive", "Moving Average"}:
            history = train["units_sold"].astype(float)
            validation_pred = _baseline_predictions(history, len(validation), model_name)
            validation_metrics = evaluate_predictions(validation["units_sold"].astype(float).to_numpy(), validation_pred.to_numpy())
            test_pred = _baseline_predictions(pd.concat([history, validation["units_sold"]], ignore_index=True), len(test), model_name)
            test_metrics = evaluate_predictions(test["units_sold"].astype(float).to_numpy(), test_pred.to_numpy())
        else:
            model = _fit_model(train, model_name=model_name)
            validation_pred = model.predict(validation[FEATURE_COLUMNS])
            validation_metrics = evaluate_predictions(validation["units_sold"].astype(float).to_numpy(), validation_pred)
            test_pred = model.predict(test[FEATURE_COLUMNS])
            test_metrics = evaluate_predictions(test["units_sold"].astype(float).to_numpy(), test_pred)
        results.append(
            {
                "model_name": model_name,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            }
        )

    selected = min(results, key=lambda item: item["validation_metrics"]["wmape"])
    return {
        "dataset_rows": len(frame),
        "feature_columns": FEATURE_COLUMNS,
        "validation_strategy": "expanding-window time-series split",
        "selected_model": selected["model_name"],
        "selected_validation_metrics": selected["validation_metrics"],
        "selected_test_metrics": selected["test_metrics"],
        "results": results,
    }


def _persist_model(model: Any, feature_columns: list[str], model_name: str, metrics: dict[str, float]) -> dict[str, Any]:
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
    }
    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, PRODUCTION_MODEL_PATH)
    return payload


def load_production_model() -> dict[str, Any]:
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError("No trained production model has been persisted yet.")
    return joblib.load(PRODUCTION_MODEL_PATH)


def train_and_select_model(df: pd.DataFrame) -> dict[str, Any]:
    experiment = run_experiment(df)
    selected_name = experiment["selected_model"]

    if selected_name in {"Naive", "Seasonal Naive", "Moving Average"}:
        artifact = {
            "model_name": selected_name,
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "feature_columns": FEATURE_COLUMNS,
            "metrics": experiment["selected_validation_metrics"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model": selected_name,
            "baseline": True,
        }
        return _persist_model(artifact["model"], artifact["feature_columns"], selected_name, artifact["metrics"])

    frame = build_feature_matrix(df).copy()
    train, _, _ = time_series_split(frame)
    model = _fit_model(train, selected_name)
    artifact = _persist_model(model, FEATURE_COLUMNS, selected_name, experiment["selected_validation_metrics"])
    artifact["selected_model"] = selected_name
    artifact["experiment"] = experiment
    return artifact


def generate_forecast_for_selection(df: pd.DataFrame, horizon: int = 14) -> list[dict[str, Any]]:
    frame = build_feature_matrix(df).copy()
    validate_no_leakage(frame)
    if frame.empty:
        raise ValueError("No data available to build the forecasting model.")

    try:
        artifact = load_production_model()
        selected_name = artifact.get("model_name")
        model = artifact.get("model")
        if selected_name is None:
            raise FileNotFoundError
    except FileNotFoundError:
        artifact = train_and_select_model(frame)
        selected_name = artifact.get("model_name")
        model = artifact.get("model")

    history = frame["units_sold"].astype(float).tolist()
    last_row = frame.iloc[-1].copy()
    last_date = pd.to_datetime(frame["date"].max())
    forecast_rows: list[dict[str, Any]] = []

    for offset in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=offset)
        feature_row = _build_future_row(history, future_date, last_row)
        if selected_name in {"Naive", "Seasonal Naive", "Moving Average"}:
            prediction = float(
                _baseline_predictions(pd.Series(history), 1, selected_name).iloc[0]
            )
        else:
            prediction = float(model.predict(pd.DataFrame([feature_row], columns=FEATURE_COLUMNS))[0])
        prediction = max(0.0, prediction)
        historical_mean = float(np.mean(history[-7:])) if history else 0.0
        residual_sigma = float(np.std(np.array(history[-30:]) - np.array(history[-30:]), ddof=0)) if len(history) >= 2 else 0.0
        lower = max(0.0, prediction - 1.96 * residual_sigma)
        upper = prediction + 1.96 * residual_sigma
        forecast_rows.append(
            {
                "date": future_date,
                "historical_demand": None,
                "forecast_demand": prediction,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
        history.append(prediction)
        last_row["units_sold"] = prediction
        last_row["lag_1"] = prediction
        last_row["lag_7"] = history[-7] if len(history) >= 7 else prediction
        last_row["lag_14"] = history[-14] if len(history) >= 14 else prediction
        last_row["rolling_mean_7"] = float(np.mean(history[-7:]))
        last_row["rolling_mean_14"] = float(np.mean(history[-14:])) if len(history) >= 14 else float(np.mean(history[-len(history):]))
        last_row["rolling_std_7"] = float(np.std(history[-7:])) if len(history) >= 2 else 0.0
        last_row["promotion"] = int(last_row.get("promotion", 0))
        last_row["holiday"] = int(last_row.get("holiday", 0))
        last_row["inventory_on_hand_lag_1"] = float(last_row.get("inventory_on_hand_lag_1", last_row.get("inventory_on_hand", 0.0)))
        last_row["lead_time_days"] = float(last_row.get("lead_time_days", last_row.get("supplier_lead_time_days", 0.0)))
        last_row["price"] = float(last_row.get("price", 0.0))
        last_row["price_change"] = float(last_row.get("price_change", 0.0))

    return forecast_rows


def compute_model_metrics(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, float]:
    frame = build_feature_matrix(df).copy()
    if len(frame) < 45:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0}

    train, validation, test = time_series_split(frame)
    model = _fit_model(train, model_name="Random Forest")
    validation_predictions = model.predict(validation[FEATURE_COLUMNS])
    metrics = evaluate_predictions(validation["units_sold"].astype(float).to_numpy(), validation_predictions)
    if not test.empty:
        test_predictions = model.predict(test[FEATURE_COLUMNS])
        metrics = evaluate_predictions(test["units_sold"].astype(float).to_numpy(), test_predictions)
    return metrics


def save_experiment_summary(experiment: dict[str, Any], path: str | Path | None = None) -> Path:
    output_path = Path(path) if path is not None else MODEL_DIR / "forecast_experiment.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(experiment, handle, indent=2, default=str)
    return output_path
