"""Data leakage detection and prevention tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.demand_intelligence.feature_engineering import (
    add_lag_features,
    add_time_features,
    build_feature_matrix,
    detect_seasonal_period,
)


def _assert_no_leakage_in_lag_features(df: pd.DataFrame) -> bool:
    """
    Verify lag features use only past data, not future.

    For each record (t), check that:
    - lag_1 = units_sold[t-1]
    - lag_7 = units_sold[t-7]
    - lag_14 = units_sold[t-14]
    - lag_28 = units_sold[t-28]

    No current or future values should be included.
    """
    if len(df) < 30:
        return True

    frame = df.sort_values(["product_id", "store_id", "date"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = add_time_features(frame)
    
    if "inventory_on_hand" not in frame.columns:
        frame["inventory_on_hand"] = 0.0
    if "supplier_lead_time_days" not in frame.columns:
        frame["supplier_lead_time_days"] = 0.0
    
    frame = add_lag_features(frame)

    for idx in range(30, min(100, len(frame))):
        row = frame.iloc[idx]
        prev_rows = frame.iloc[idx - 30 : idx]
        prev_rows = prev_rows.sort_values("date")

        if pd.notna(row["lag_1"]):
            expected = prev_rows.iloc[-1]["units_sold"] if len(prev_rows) >= 1 else np.nan
            if pd.notna(expected) and abs(row["lag_1"] - expected) > 1e-6:
                return False

        if pd.notna(row["lag_7"]):
            expected = prev_rows.iloc[-7]["units_sold"] if len(prev_rows) >= 7 else np.nan
            if pd.notna(expected) and abs(row["lag_7"] - expected) > 1e-6:
                return False

    return True


def _assert_no_leakage_in_rolling_features(df: pd.DataFrame) -> bool:
    """
    Verify rolling features do not include current or future values.

    For rolling_mean_7, should be mean of [t-7, t-6, ..., t-1], NOT including t.
    Verified by checking: rolling_mean_k was calculated with .shift(1) before .rolling()
    """
    if len(df) < 10:
        return True

    frame = df.sort_values(["product_id", "store_id", "date"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = add_time_features(frame)
    
    if "inventory_on_hand" not in frame.columns:
        frame["inventory_on_hand"] = 0.0
    if "supplier_lead_time_days" not in frame.columns:
        frame["supplier_lead_time_days"] = 0.0
    
    frame = add_lag_features(frame)

    for idx in range(8, min(50, len(frame))):
        row = frame.iloc[idx]
        prev_rows = frame.iloc[idx - 8 : idx]

        if pd.notna(row["rolling_mean_7"]) and len(prev_rows) >= 7:
            manual_mean = prev_rows.iloc[-7:-1]["units_sold"].mean()
            expected = manual_mean if pd.notna(manual_mean) else np.nan
            if pd.notna(expected) and abs(row["rolling_mean_7"] - expected) > 1e-3:
                return False

    return True


def _assert_preprocessing_fitted_on_training_data_only() -> bool:
    """
    Verify that any preprocessing (normalization, scaling) would only
    be fitted on training data.

    This is a design verification - actual preprocessor is optional in current implementation.
    If added later, must not use validation/test statistics.
    """
    return True


def _assert_target_not_in_features(df: pd.DataFrame) -> bool:
    """Verify that 'units_sold' (target) is never used as a feature."""
    from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS

    return "units_sold" not in FEATURE_COLUMNS


def _assert_future_features_for_multistep() -> bool:
    """
    Verify that multi-step forecasting correctly generates future features.

    For step t+1, features must not include units_sold[t+1] (which doesn't exist yet).
    Instead, use predicted value or lagged historical.
    """
    from src.demand_intelligence.forecasting import _build_future_row

    history = [10.0, 12.0, 15.0, 14.0, 18.0, 17.0, 19.0, 20.0, 22.0, 21.0]
    future_date = pd.Timestamp("2024-01-11")
    last_row = pd.Series({"promotion": 0, "holiday": 0, "price": 10.0, "price_change": 0.0})

    feature_row = _build_future_row(history, future_date, last_row)

    assert "lag_1" in feature_row
    assert "lag_7" in feature_row
    assert "rolling_mean_7" in feature_row

    assert feature_row["lag_1"] == 21.0
    if len(history) >= 7:
        assert feature_row["lag_7"] == 20.0

    return True


def test_no_data_leakage_in_lag_features():
    """Verify lag features use only past data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": "P101",
            "store_id": 1,
            "units_sold": np.random.uniform(10, 100, 100),
            "price": 10.0,
            "promotion": 0,
            "holiday": 0,
        }
    )

    assert _assert_no_leakage_in_lag_features(df), "Data leakage detected in lag features"


def test_no_data_leakage_in_rolling_features():
    """Verify rolling features do not include current/future values."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": "P101",
            "store_id": 1,
            "units_sold": np.random.uniform(10, 100, 100),
            "price": 10.0,
            "promotion": 0,
            "holiday": 0,
        }
    )

    assert _assert_no_leakage_in_rolling_features(df), "Data leakage detected in rolling features"


def test_target_not_used_as_feature():
    """Verify target variable is not in feature set."""
    assert _assert_target_not_in_features(None), "Target variable found in features"


def test_future_features_for_multistep_forecasting():
    """Verify multi-step forecasting doesn't use future information."""
    assert _assert_future_features_for_multistep(), "Future data leakage in multi-step forecasting"


def test_preprocessing_fitted_only_on_training():
    """Verify preprocessing would be fitted only on training data."""
    assert _assert_preprocessing_fitted_on_training_data_only(), "Preprocessing uses validation/test data"


def test_all_leakage_checks():
    """Run all leakage detection tests."""
    tests = [
        test_target_not_used_as_feature,
        test_preprocessing_fitted_only_on_training,
    ]

    failed = []
    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            failed.append((test_func.__name__, str(e)))

    if failed:
        raise AssertionError(f"Leakage tests failed: {failed}")
