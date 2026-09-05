import numpy as np
import pandas as pd
import pytest

from src.demand_intelligence.forecast_analytics import (
    build_forecast_accuracy_analytics,
    build_forecast_error_dataset,
    calculate_accuracy_metrics,
)


class ConstantModel:
    def __init__(self, value=12.0):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def make_frame(days=28):
    dates = pd.date_range("2024-01-01", periods=days)
    return pd.DataFrame({
        "date": dates,
        "product_id": ["P1"] * days,
        "product_name": ["One"] * days,
        "store_id": [1] * days,
        "region": ["North"] * days,
        "category": ["Home"] * days,
        "units_sold": np.arange(10, 10 + days, dtype=float),
        "price": [20.0] * days,
        "promotion": [0] * days,
        "holiday": [0] * days,
        "inventory_on_hand": [50.0] * days,
        "supplier_lead_time_days": [5] * days,
    })


def test_error_calculation():
    frame, _ = build_forecast_error_dataset(make_frame(), ConstantModel(), ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "day_of_week", "month", "week_of_year", "day_of_year", "quarter", "is_weekend", "promotion", "holiday", "price", "price_change", "inventory_on_hand_lag_1", "lead_time_days"])
    assert frame.iloc[0]["error"] == pytest.approx(12 - 24)


def test_absolute_error():
    metrics = calculate_accuracy_metrics([10], [14])
    assert metrics["mae"] == 4


def test_squared_error():
    frame, _ = build_forecast_error_dataset(make_frame(), ConstantModel(), None)
    assert frame.iloc[0]["squared_error"] == pytest.approx(frame.iloc[0]["error"] ** 2)


def test_zero_actual_mape_safe():
    metrics = calculate_accuracy_metrics([0, 10], [5, 10])
    assert np.isfinite(metrics["mape"])


def test_nan_metric_rejected():
    with pytest.raises(ValueError):
        calculate_accuracy_metrics([1, np.nan], [1, 2])


def test_infinity_metric_rejected():
    with pytest.raises(ValueError):
        calculate_accuracy_metrics([1, np.inf], [1, 2])


def test_nan_rows_are_excluded():
    data = make_frame()
    data.loc[0, "units_sold"] = np.nan
    _, quality = build_forecast_error_dataset(data, ConstantModel())
    assert quality["input_rows"] == len(data)
    assert quality["excluded_rows"] >= 1


def test_invalid_date_is_excluded():
    data = make_frame()
    data["date"] = data["date"].astype(object)
    data.loc[0, "date"] = "not-a-date"
    _, quality = build_forecast_error_dataset(data, ConstantModel())
    assert quality["excluded_rows"] >= 1


def test_nonfinite_prediction_rejected():
    with pytest.raises(ValueError):
        build_forecast_error_dataset(make_frame(), ConstantModel(np.inf))


def test_mae():
    assert calculate_accuracy_metrics([1, 3], [2, 5])["mae"] == pytest.approx(1.5)


def test_rmse():
    assert calculate_accuracy_metrics([1, 3], [2, 5])["rmse"] == pytest.approx(np.sqrt(2.5))


def test_mape():
    assert calculate_accuracy_metrics([10, 20], [11, 18])["mape"] == pytest.approx(10)


def test_wmape():
    assert calculate_accuracy_metrics([10, 20], [11, 18])["wmape"] == pytest.approx(10)


def test_bias_positive_for_overforecast():
    assert calculate_accuracy_metrics([10], [12])["bias"] == pytest.approx(0.2)


def test_bias_negative_for_underforecast():
    assert calculate_accuracy_metrics([10], [8])["bias"] == pytest.approx(-0.2)


def test_over_forecast_classification():
    data = make_frame()
    data["units_sold"] = 1
    frame, _ = build_forecast_error_dataset(data, ConstantModel(2))
    assert set(frame["classification"]) == {"OVER_FORECAST"}


def test_under_forecast_classification():
    data = make_frame()
    data["units_sold"] = 20
    frame, _ = build_forecast_error_dataset(data, ConstantModel(2))
    assert set(frame["classification"]) == {"UNDER_FORECAST"}


def test_exact_classification():
    data = make_frame()
    data["units_sold"] = 12
    frame, _ = build_forecast_error_dataset(data, ConstantModel(12))
    assert set(frame["classification"]) == {"EXACT"}


def test_summary_rates():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert 0 <= result["summary"]["over_forecast_rate"] <= 1
    assert 0 <= result["summary"]["under_forecast_rate"] <= 1


def test_bias_label():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert result["bias"]["label"] in {"OVER_FORECAST", "UNDER_FORECAST", "BALANCED"}


def test_product_aggregation():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert len(result["breakdowns"]["product"]) == 1
    assert result["breakdowns"]["product"][0]["product_id"] == "P1"


def test_product_metrics():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert "wmape" in result["breakdowns"]["product"][0]


def test_product_ranking():
    data = pd.concat([make_frame(), make_frame().assign(product_id="P2", product_name="Two")], ignore_index=True)
    result = build_forecast_accuracy_analytics(data, model=ConstantModel())
    assert result["best_worst"]["best_product"]["product_id"] in {"P1", "P2"}


def test_minimum_observation_status():
    result = build_forecast_accuracy_analytics(make_frame(21), model=ConstantModel(), minimum_observations=20)
    assert result["breakdowns"]["product"][0]["status"] == "INSUFFICIENT_DATA"


def test_store_aggregation():
    data = pd.concat([make_frame(), make_frame().assign(store_id=2)], ignore_index=True)
    assert len(build_forecast_accuracy_analytics(data, model=ConstantModel())["breakdowns"]["store"]) == 2


def test_store_ranking():
    data = pd.concat([make_frame(), make_frame().assign(store_id=2)], ignore_index=True)
    assert build_forecast_accuracy_analytics(data, model=ConstantModel())["best_worst"]["worst_store"]


def test_category_aggregation():
    data = pd.concat([make_frame(), make_frame().assign(category="Beauty")], ignore_index=True)
    assert len(build_forecast_accuracy_analytics(data, model=ConstantModel())["breakdowns"]["category"]) == 2


def test_category_bias():
    data = pd.concat([make_frame(), make_frame().assign(category="Beauty", units_sold=5)], ignore_index=True)
    result = build_forecast_accuracy_analytics(data, model=ConstantModel())
    assert all("bias" in item for item in result["breakdowns"]["category"])


def test_region_aggregation():
    data = pd.concat([make_frame(), make_frame().assign(region="South")], ignore_index=True)
    assert len(build_forecast_accuracy_analytics(data, model=ConstantModel())["breakdowns"]["region"]) == 2


def test_region_bias():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert "bias" in result["breakdowns"]["region"][0]


def test_daily_trend():
    assert build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())["trends"]["day"]


def test_weekly_trend():
    assert build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())["trends"]["week"]


def test_monthly_trend():
    assert build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())["trends"]["month"]


def test_underforecast_units():
    data = make_frame()
    data["units_sold"] = 20
    result = build_forecast_accuracy_analytics(data, model=ConstantModel(10))
    assert result["business_impact"]["under_forecast_units"] > 0


def test_overforecast_units():
    data = make_frame()
    data["units_sold"] = 5
    result = build_forecast_accuracy_analytics(data, model=ConstantModel(10))
    assert result["business_impact"]["over_forecast_units"] > 0


def test_stockout_indicator():
    data = make_frame()
    data["inventory_on_hand"] = 1
    assert build_forecast_accuracy_analytics(data, model=ConstantModel())["business_impact"]["stockout_risk_count"] > 0


def test_excess_indicator():
    data = make_frame()
    data["inventory_on_hand"] = 100
    assert build_forecast_accuracy_analytics(data, model=ConstantModel())["business_impact"]["excess_inventory_risk_count"] > 0


def test_metadata_reports_exclusions():
    data = make_frame()
    data.loc[0, "units_sold"] = np.nan
    result = build_forecast_accuracy_analytics(data, model=ConstantModel())
    assert result["metadata"]["excluded_rows"] >= 1


def test_empty_data_fails():
    with pytest.raises(ValueError):
        build_forecast_accuracy_analytics(pd.DataFrame(columns=make_frame().columns), model=ConstantModel())


def test_missing_columns_fail():
    with pytest.raises(ValueError):
        build_forecast_error_dataset(pd.DataFrame({"date": []}), ConstantModel())


def test_mismatched_metric_lengths_fail():
    with pytest.raises(ValueError):
        calculate_accuracy_metrics([1], [1, 2])


def test_threshold_validation():
    with pytest.raises(ValueError):
        build_forecast_accuracy_analytics(make_frame(), model=ConstantModel(), minimum_observations=0)


def test_bias_threshold_validation():
    with pytest.raises(ValueError):
        build_forecast_accuracy_analytics(make_frame(), model=ConstantModel(), bias_threshold=-1)


def test_all_output_numbers_are_finite():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert all(np.isfinite(value) for value in result["summary"].values() if isinstance(value, (int, float)))


def test_observation_count_matches():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert result["summary"]["observation_count"] == len(make_frame()) - 14


def test_operational_impact_is_not_monetary():
    result = build_forecast_accuracy_analytics(make_frame(), model=ConstantModel())
    assert result["business_impact"]["interpretation"].startswith("Operational indicators")
