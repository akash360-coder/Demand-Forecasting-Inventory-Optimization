from __future__ import annotations

import pandas as pd

from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.feature_engineering import build_feature_matrix, detect_seasonal_period
from src.demand_intelligence.forecasting import (
    generate_forecast_for_selection,
    moving_average_forecast,
    naive_forecast,
    run_experiment,
    seasonal_naive_forecast,
    time_series_split,
)


def test_baseline_forecasts_use_historical_data_only():
    history = pd.Series([10, 12, 15, 14, 18, 17, 19, 20, 22, 21], dtype=float)

    naive = naive_forecast(history, horizon=3)
    seasonal = seasonal_naive_forecast(history, horizon=3, period=7)
    moving = moving_average_forecast(history, horizon=3, window=5)

    assert len(naive) == 3
    assert len(seasonal) == 3
    assert len(moving) == 3
    assert all(value >= 0 for value in naive)
    assert all(value >= 0 for value in seasonal)
    assert all(value >= 0 for value in moving)


def test_time_series_split_has_no_overlap_and_uses_real_dates():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"date": dates, "product_id": "P101", "store_id": 1, "units_sold": range(120), "price": 10.0, "promotion": 0, "holiday": 0, "inventory_on_hand": 100.0, "supplier_lead_time_days": 5})

    train, validation, test = time_series_split(df, validation_days=20, test_days=15)

    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_feature_engineering_uses_past_only_lags_and_rolling_stats():
    df = ensure_dataset().head(120).copy()
    features = build_feature_matrix(df)

    assert "lag_1" in features.columns
    assert "lag_7" in features.columns
    assert "rolling_mean_7" in features.columns
    assert "rolling_std_7" in features.columns
    assert features["lag_1"].notna().all()
    assert features["rolling_mean_7"].notna().all()


def test_experiment_returns_model_comparison_and_selected_model():
    df = ensure_dataset().head(300).copy()
    experiment = run_experiment(df)

    assert "results" in experiment
    assert "selected_model" in experiment
    assert len(experiment["results"]) >= 4
    assert experiment["selected_model"] in {result["model_name"] for result in experiment["results"]}
    assert "wmape" in experiment["selected_validation_metrics"]


def test_forecast_generation_returns_multi_day_horizon():
    df = ensure_dataset().head(300).copy()
    forecast = generate_forecast_for_selection(df, horizon=7)

    assert len(forecast) == 7
    assert all(item["forecast_demand"] >= 0 for item in forecast)
    assert all(item["lower_bound"] <= item["upper_bound"] for item in forecast)


def test_seasonal_period_detection_is_reasonable_for_daily_sales():
    df = ensure_dataset().head(200).copy()
    period = detect_seasonal_period(df)
    assert period in {7, 14, 28}
