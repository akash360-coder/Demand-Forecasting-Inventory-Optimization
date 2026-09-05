import os

os.environ.setdefault("DATABASE_MODE", "sqlite")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.monitoring_service import get_monitoring_response
from src.demand_intelligence.monitoring import categorical_drift, data_quality, drift_status, performance_comparison, psi


client = TestClient(app)


def test_monitoring_api_returns_real_payload():
    response = client.get("/api/v1/monitoring")
    assert response.status_code == 200
    assert {"status", "data_quality", "feature_drift", "target_drift", "prediction_drift", "model_performance", "alerts", "metadata"} <= response.json().keys()


@pytest.mark.parametrize("values", [[1, 2, 3], [0, 0, 0], [1, np.nan, 2], [1, np.inf, 2], [1, -np.inf, 2]])
def test_psi_is_finite(values):
    assert np.isfinite(psi([1, 2, 3], values))


def test_identical_distribution_has_near_zero_psi():
    assert psi([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0)


def test_shifted_distribution_has_positive_psi():
    assert psi([1, 2, 3, 4], [10, 11, 12, 13]) > 0


@pytest.mark.parametrize(("score", "expected"), [(0.01, "HEALTHY"), (0.1, "WARNING"), (0.3, "CRITICAL")])
def test_drift_threshold_status(score, expected):
    assert drift_status(score) == expected


def test_categorical_identical_distribution_is_zero():
    assert categorical_drift(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(0)


def test_categorical_shift_is_positive():
    assert categorical_drift(["a", "a", "a"], ["b", "b", "a"]) > 0


def test_quality_reports_missing_values():
    frame = pd.DataFrame({"date": ["2024-01-01"], "price": [np.nan], "promotion": [0]})
    result = data_quality(frame)
    assert result["missing"]["price"]["missing_count"] == 1
    assert result["status"] == "CRITICAL"


def test_quality_reports_duplicates():
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "price": [1, 1]})
    assert data_quality(frame)["duplicate_count"] == 1


def test_quality_reports_invalid_values():
    frame = pd.DataFrame({"date": ["2024-01-01"], "price": [-1], "inventory_on_hand": [-2], "supplier_lead_time_days": [0], "promotion": [2], "holiday": [-1], "units_sold": [-3]})
    result = data_quality(frame)
    assert result["invalid_count"] == 6
    assert result["status"] == "CRITICAL"


def test_quality_reports_non_finite_values():
    frame = pd.DataFrame({"date": ["2024-01-01"], "price": [np.inf]})
    assert data_quality(frame)["non_finite_count"] == 1


def test_quality_empty_dataset_is_safe():
    result = data_quality(pd.DataFrame(columns=["date", "price"]))
    assert result["row_count"] == 0
    assert result["duplicate_rate"] == 0


def test_quality_reports_date_coverage():
    result = data_quality(pd.DataFrame({"date": ["2024-01-01", "2024-01-03"]}))
    assert result["date_min"] == "2024-01-01"
    assert result["date_max"] == "2024-01-03"


def test_performance_metrics_are_reused():
    result = performance_comparison([1, 2, 3], [1, 3, 3], {"mae": 0, "rmse": 0, "mape": 0, "wmape": 0})
    assert {item["metric"] for item in result} == {"MAE", "RMSE", "MAPE", "WMAPE"}


def test_performance_degradation_status():
    result = performance_comparison([1, 2, 3], [10, 10, 10], {"mae": 1, "rmse": 1, "mape": 1, "wmape": 1})
    assert all(item["status"] == "CRITICAL" for item in result)


def test_monitoring_uses_production_features():
    result = get_monitoring_response()
    assert len(result.feature_drift) == 18
    assert all(np.isfinite(item["score"]) for item in result.feature_drift)


def test_monitoring_metadata_has_periods():
    result = get_monitoring_response()
    assert result.metadata["reference_period"]
    assert result.metadata["monitoring_period"]


def test_monitoring_status_is_valid():
    assert get_monitoring_response().status in {"HEALTHY", "WARNING", "CRITICAL"}


def test_monitoring_scores_are_finite():
    result = get_monitoring_response()
    assert np.isfinite(result.target_drift["score"])
    assert np.isfinite(result.prediction_drift["score"])


def test_monitoring_filter_accepts_real_product():
    assert get_monitoring_response(product_id="P100", store_id=1).metadata["monitoring_rows"] > 0


def test_monitoring_rejects_empty_window():
    with pytest.raises(ValueError):
        get_monitoring_response(start_date="2099-01-01")


def test_monitoring_api_rejects_invalid_store_filter():
    assert client.get("/api/v1/monitoring", params={"store_id": 0}).status_code == 422


def test_monitoring_api_rejects_empty_window():
    assert client.get("/api/v1/monitoring", params={"start_date": "2099-01-01"}).status_code == 400


def test_alerts_have_required_fields():
    for alert in get_monitoring_response().alerts:
        assert {"severity", "category", "message", "metric", "score"} <= alert.keys()


def test_feature_statuses_are_valid():
    assert all(item["status"] in {"HEALTHY", "WARNING", "CRITICAL"} for item in get_monitoring_response().feature_drift)


def test_target_status_is_valid():
    assert get_monitoring_response().target_drift["status"] in {"HEALTHY", "WARNING", "CRITICAL"}


def test_prediction_status_is_valid():
    assert get_monitoring_response().prediction_drift["status"] in {"HEALTHY", "WARNING", "CRITICAL"}
