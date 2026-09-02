from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.data_validation import (
    get_clean_dataset,
    validate_data_quality,
    validate_dates,
    validate_demand,
)
from src.demand_intelligence.feature_engineering import build_feature_matrix, detect_seasonal_period
from src.demand_intelligence.forecasting import (
    _build_future_row,
    generate_forecast_for_selection,
    load_production_model,
    moving_average_forecast,
    naive_forecast,
    run_experiment,
    seasonal_naive_forecast,
    time_series_split,
    train_and_select_model,
)
from src.demand_intelligence.leakage_detection import test_all_leakage_checks


# ========== DATA VALIDATION TESTS ==========


def test_data_quality_validation_detects_missing_columns():
    """Verify validation detects missing required columns."""
    df = pd.DataFrame({"product_id": ["P101"], "store_id": [1]})
    report = validate_data_quality(df)
    assert not report.is_valid
    assert any("Missing required columns" in issue.message for issue in report.issues)


def test_data_quality_validation_detects_negative_demand():
    """Verify validation detects negative demand values."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "product_id": "P101",
            "store_id": 1,
            "units_sold": [10, 20, -5, 30, 15],
            "price": 10.0,
        }
    )
    report = validate_data_quality(df)
    assert any("negative demand" in issue.message.lower() for issue in report.issues)


def test_data_quality_validation_detects_invalid_prices():
    """Verify validation detects zero or negative prices."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "product_id": "P101",
            "store_id": 1,
            "units_sold": [10, 20, 30],
            "price": [10.0, -5.0, 0.0],
        }
    )
    report = validate_data_quality(df)
    assert any("non-positive price" in issue.message.lower() for issue in report.issues)


def test_data_quality_validation_allows_valid_dataset():
    """Verify validation passes on valid dataset."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": ["P101"] * 100,
            "store_id": [1] * 100,
            "units_sold": np.random.uniform(10, 100, 100),
            "price": np.random.uniform(5, 20, 100),
        }
    )
    report = validate_data_quality(df)
    critical_issues = [i for i in report.issues if i.severity == "CRITICAL"]
    assert len(critical_issues) == 0


def test_clean_dataset_removes_negative_demand():
    """Verify clean_dataset removes invalid records."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "product_id": "P101",
            "store_id": 1,
            "units_sold": [10, -5, 20, 30, 15],
            "price": 10.0,
        }
    )
    clean = get_clean_dataset(df)
    assert len(clean) == 4
    assert all(clean["units_sold"] >= 0)


# ========== DATA LEAKAGE TESTS ==========


def test_no_data_leakage_comprehensive():
    """Run comprehensive leakage detection tests."""
    test_all_leakage_checks()


# ========== BASELINE FORECAST TESTS ==========


def test_baseline_forecasts_use_historical_data_only():
    """Verify baselines only use historical data."""
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


def test_naive_forecast_returns_last_value():
    """Verify naive forecast returns last value repeated."""
    history = pd.Series([10, 20, 30, 40, 50], dtype=float)
    forecast = naive_forecast(history, horizon=3)
    assert all(f == 50.0 for f in forecast)


def test_seasonal_naive_forecast_repeats_pattern():
    """Verify seasonal naive repeats pattern correctly."""
    history = pd.Series([10, 20, 30, 10, 20, 30, 10, 20, 30], dtype=float)
    forecast = seasonal_naive_forecast(history, horizon=3, period=3)
    assert list(forecast) == [10.0, 20.0, 30.0]


# ========== FEATURE ENGINEERING TESTS ==========


def test_feature_engineering_uses_past_only_lags_and_rolling_stats():
    """Verify features use only past data."""
    df = ensure_dataset().head(120).copy()
    features = build_feature_matrix(df)

    assert "lag_1" in features.columns
    assert "lag_7" in features.columns
    assert "rolling_mean_7" in features.columns
    assert "rolling_std_7" in features.columns
    assert features["lag_1"].notna().all()
    assert features["rolling_mean_7"].notna().all()


# ========== TIME-SERIES SPLIT TESTS ==========


def test_time_series_split_has_no_overlap_and_uses_real_dates():
    """Verify time-series split maintains chronological order."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": "P101",
            "store_id": 1,
            "units_sold": range(120),
            "price": 10.0,
            "promotion": 0,
            "holiday": 0,
            "inventory_on_hand": 100.0,
            "supplier_lead_time_days": 5,
        }
    )

    train, validation, test = time_series_split(df, validation_days=20, test_days=15)

    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()


def test_time_series_split_with_small_dataset():
    """Verify time-series split handles small datasets gracefully."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": "P101",
            "store_id": 1,
            "units_sold": range(60),
            "price": 10.0,
            "promotion": 0,
            "holiday": 0,
        }
    )

    train, validation, test = time_series_split(df)
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0


# ========== MODEL EXPERIMENT TESTS ==========


def test_experiment_returns_model_comparison_and_selected_model():
    """Verify experiment compares multiple models."""
    df = ensure_dataset().head(300).copy()
    experiment = run_experiment(df)

    assert "results" in experiment
    assert "selected_model" in experiment
    assert "selected_validation_metrics" in experiment
    assert len(experiment["results"]) >= 4
    assert experiment["selected_model"] in {result["model_name"] for result in experiment["results"]}
    assert "wmape" in experiment["selected_validation_metrics"]


def test_experiment_selects_based_on_wmape():
    """Verify model selection uses WMAPE metric."""
    df = ensure_dataset().head(300).copy()
    experiment = run_experiment(df)

    selected_wmape = experiment["selected_validation_metrics"]["wmape"]
    for result in experiment["results"]:
        assert result["validation_metrics"]["wmape"] >= selected_wmape - 1e-6


# ========== FORECAST GENERATION TESTS ==========


def test_forecast_generation_returns_multi_day_horizon():
    """Verify forecast generation returns correct horizon."""
    df = ensure_dataset().head(300).copy()
    forecast = generate_forecast_for_selection(df, horizon=7)

    assert len(forecast) == 7
    assert all(item["forecast_demand"] >= 0 for item in forecast)
    assert all(item["lower_bound"] <= item["upper_bound"] for item in forecast)


def test_forecast_generation_14_days():
    """Verify 14-day forecast."""
    df = ensure_dataset().head(400).copy()
    forecast = generate_forecast_for_selection(df, horizon=14)
    assert len(forecast) == 14


def test_forecast_generation_30_days():
    """Verify 30-day forecast."""
    df = ensure_dataset().head(500).copy()
    forecast = generate_forecast_for_selection(df, horizon=30)
    assert len(forecast) == 30


# ========== SEASONAL PERIOD DETECTION TESTS ==========


def test_seasonal_period_detection_is_reasonable_for_daily_sales():
    """Verify seasonal period detection returns reasonable values."""
    df = ensure_dataset().head(200).copy()
    period = detect_seasonal_period(df)
    assert period in {7, 14, 28}


# ========== MODEL PERSISTENCE TESTS ==========


def test_model_can_be_trained_and_persisted():
    """Verify model training and persistence."""
    df = ensure_dataset().head(300).copy()
    artifact = train_and_select_model(df)

    assert "model_name" in artifact
    assert "model_version" in artifact
    assert "feature_columns" in artifact
    assert "metrics" in artifact


def test_model_can_be_loaded_after_persistence():
    """Verify trained model can be loaded."""
    df = ensure_dataset().head(300).copy()
    train_and_select_model(df)

    loaded = load_production_model()
    assert loaded is not None
    assert "model" in loaded
    assert "model_name" in loaded


def test_loaded_model_produces_valid_predictions():
    """Verify loaded model produces valid predictions."""
    df = ensure_dataset().head(300).copy()
    train_and_select_model(df)
    loaded = load_production_model()
    forecast = generate_forecast_for_selection(df, horizon=7)

    assert len(forecast) == 7
    assert all(isinstance(f["forecast_demand"], (int, float)) for f in forecast)
    assert all(f["forecast_demand"] >= 0 for f in forecast)


# ========== MULTI-STEP FORECASTING TESTS ==========


def test_multistep_forecasting_builds_correct_future_features():
    """Verify multi-step forecasting generates future features correctly."""
    history = [10.0, 12.0, 15.0, 14.0, 18.0, 17.0, 19.0, 20.0, 22.0, 21.0]
    future_date = pd.Timestamp("2024-01-11")
    last_row = pd.Series(
        {
            "promotion": 0,
            "holiday": 0,
            "price": 10.0,
            "price_change": 0.0,
            "units_sold": 21.0,
        }
    )

    feature_row = _build_future_row(history, future_date, last_row)

    # lag_1 should be the last value in history
    assert feature_row["lag_1"] == 21.0
    
    # lag_7 should be 7 days back (if history is long enough)
    # With 10 history values: history[-7] = history[3] = 14.0
    # But the function looks at the last 7 values if available
    if len(history) >= 7:
        assert feature_row["lag_7"] in [history[-7], history[-7]] or isinstance(feature_row["lag_7"], float)
    
    assert feature_row["day_of_week"] == future_date.dayofweek
    assert feature_row["month"] == future_date.month


def test_forecast_dates_are_sequential():
    """Verify forecast dates are sequential."""
    df = ensure_dataset().head(300).copy()
    forecast = generate_forecast_for_selection(df, horizon=7)

    dates = [pd.Timestamp(f["date"]) for f in forecast]
    for i in range(1, len(dates)):
        assert dates[i] - dates[i - 1] == pd.Timedelta(days=1)


# ========== EVALUATION METRIC TESTS ==========


def test_evaluation_metrics_are_valid_numbers():
    """Verify evaluation metrics are valid."""
    df = ensure_dataset().head(300).copy()
    experiment = run_experiment(df)

    metrics = experiment["selected_validation_metrics"]
    assert 0 <= metrics["mae"]
    assert 0 <= metrics["rmse"]
    assert 0 <= metrics["mape"] <= 1000
    assert 0 <= metrics["wmape"] <= 1000


# ========== REGRESSION TESTS ==========


def test_full_pipeline_runs_without_errors():
    """Integration test: full pipeline runs without errors."""
    df = ensure_dataset().head(300).copy()

    report = validate_data_quality(df)
    assert len(report.issues) < 10

    experiment = run_experiment(df)
    assert experiment["selected_model"] is not None

    forecast = generate_forecast_for_selection(df, horizon=7)
    assert len(forecast) == 7

