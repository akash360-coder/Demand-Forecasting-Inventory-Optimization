import pandas as pd
import pytest

from src.demand_intelligence.feature_engineering import (
    FEATURE_COLUMNS,
    add_lag_features,
    build_feature_matrix,
)
from src.demand_intelligence.forecasting import _build_future_row, time_series_split
from src.demand_intelligence.leakage_detection import (
    validate_chronological_order,
    validate_no_leakage,
)


def _frame(values, product="P1", store=1):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(values), freq="D"),
            "product_id": product,
            "store_id": store,
            "units_sold": values,
            "price": 10.0,
            "promotion": 0,
            "holiday": 0,
            "inventory_on_hand": 100.0,
            "supplier_lead_time_days": 5,
        }
    )


def test_target_not_in_feature_matrix():
    features = build_feature_matrix(_frame(range(40)))
    validate_no_leakage(features)
    assert "units_sold" not in FEATURE_COLUMNS


def test_lag_and_rolling_features_use_only_history():
    features = add_lag_features(_frame([10, 20, 30, 40] + list(range(50))))
    row = features.loc[features["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert row["lag_1"] == 20
    assert row["rolling_mean_7"] == 15


def test_group_lags_do_not_cross_entities():
    first = _frame(range(40), "P1", 1)
    second = _frame([100 + i for i in range(40)], "P2", 2)
    features = build_feature_matrix(pd.concat([first, second], ignore_index=True))
    row = features[(features.product_id == "P2") & (features.date == "2024-01-30")].iloc[0]
    assert row["lag_1"] == 128


def test_chronological_order_is_required():
    frame = _frame(range(5)).iloc[[1, 0, 2, 3, 4]]
    with pytest.raises(ValueError, match="chronologically ordered"):
        validate_chronological_order(frame)


def test_split_partitions_are_chronological():
    frame = build_feature_matrix(_frame(range(100)))
    train, validation, test = time_series_split(frame)
    assert train.date.max() < validation.date.min() < test.date.min()


def test_recursive_row_uses_predictions_history_only():
    history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    row = _build_future_row(
        history,
        pd.Timestamp("2024-01-08"),
        pd.Series({"units_sold": 9999.0}),
    )
    assert row["lag_1"] == 70.0
    assert row["rolling_mean_7"] == sum(history) / 7
