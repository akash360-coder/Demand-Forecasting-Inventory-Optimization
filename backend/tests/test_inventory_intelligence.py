import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DATABASE_MODE", "sqlite")

import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.demand_intelligence.inventory_intelligence import (
    abc_xyz_matrix,
    classify_abc,
    classify_xyz,
    excess_inventory_intelligence,
    inventory_health_band,
    inventory_health_score,
    opportunity_detection,
    service_level_analysis,
    stockout_risk_intelligence,
)

client = TestClient(app)


def make_inventory_frame(rows=14):
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame({
        "date": dates,
        "product_id": "P1",
        "store_id": 1,
        "category": "Home",
        "region": "North",
        "units_sold": [10, 12, 11, 13, 12, 10, 9, 11, 12, 10, 8, 9, 10, 11],
        "price": [20.0] * rows,
        "inventory_on_hand": [50.0] * rows,
        "supplier_lead_time_days": [5.0] * rows,
    })


def test_abc_classification_returns_expected_shape():
    rows = classify_abc(pd.DataFrame({"product_id": ["P1", "P2", "P3"], "business_value": [100, 60, 10]}))
    assert rows[0]["abc_class"] == "A"
    assert rows[0]["percentage_of_total"] > 0
    assert all("product_id" in item for item in rows)


def test_abc_threshold_behavior():
    rows = classify_abc(pd.DataFrame({"product_id": ["P1", "P2", "P3"], "business_value": [80, 15, 5]}))
    assert rows[0]["abc_class"] == "A"
    assert rows[1]["abc_class"] == "B"
    assert rows[2]["abc_class"] == "C"


def test_abc_ties_are_deterministic():
    rows = classify_abc(pd.DataFrame({"product_id": ["P2", "P1"], "business_value": [100, 100]}))
    assert rows[0]["product_id"] == "P1"
    assert rows[1]["product_id"] == "P2"


def test_abc_zero_value_is_safe():
    rows = classify_abc(pd.DataFrame({"product_id": ["P1", "P2"], "business_value": [0, 0]}))
    assert rows and all(item["percentage_of_total"] == 0 for item in rows)
    assert all(item["abc_class"] == "C" for item in rows)


def test_abc_empty_dataset_returns_empty():
    assert classify_abc(pd.DataFrame(columns=["product_id", "business_value"])) == []


def test_xyz_x_classification():
    rows = classify_xyz(pd.DataFrame({"product_id": ["P1", "P1", "P1", "P1", "P1"], "units_sold": [10, 11, 9, 10, 10]}))
    assert rows[0]["xyz_class"] == "X"


def test_xyz_y_classification():
    rows = classify_xyz(pd.DataFrame({"product_id": ["P1"] * 5, "units_sold": [10, 15, 12, 8, 13]}))
    assert rows[0]["xyz_class"] == "Y"


def test_xyz_z_classification():
    rows = classify_xyz(pd.DataFrame({"product_id": ["P1"] * 5, "units_sold": [1, 0, 8, 0, 20]}))
    assert rows[0]["xyz_class"] == "Z"


def test_xyz_zero_mean_is_safe():
    rows = classify_xyz(pd.DataFrame({"product_id": ["P1", "P1"], "units_sold": [0, 0]}))
    assert rows[0]["coefficient_of_variation"] == 0
    assert rows[0]["xyz_class"] == "X"


def test_xyz_zero_variance_is_x():
    rows = classify_xyz(pd.DataFrame({"product_id": ["P1"] * 4, "units_sold": [5, 5, 5, 5]}))
    assert rows[0]["xyz_class"] == "X"


def test_xyz_missing_data_is_safe():
    frame = pd.DataFrame({"product_id": ["P1", "P1", "P1"], "units_sold": [None, 2, 4]})
    rows = classify_xyz(frame)
    assert rows[0]["mean_demand"] >= 0
    assert rows[0]["demand_std"] >= 0


def test_xyz_deterministic_behavior():
    first = classify_xyz(pd.DataFrame({"product_id": ["P1"] * 4, "units_sold": [1, 2, 3, 4]}))
    second = classify_xyz(pd.DataFrame({"product_id": ["P1"] * 4, "units_sold": [1, 2, 3, 4]}))
    assert first == second


def test_abc_xyz_matrix_has_all_segments():
    abc = classify_abc(pd.DataFrame({"product_id": ["P1", "P2", "P3"], "business_value": [100, 60, 10]}))
    xyz = classify_xyz(pd.DataFrame({"product_id": ["P1", "P2", "P3"], "units_sold": [9, 12, 20]}))
    matrix, _ = abc_xyz_matrix(abc, xyz)
    assert set(matrix) >= {"AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"}


def test_abc_xyz_matrix_missing_class_is_ignored():
    abc = [{"product_id": "P1", "business_value": 100, "abc_class": "A"}]
    xyz = [{"product_id": "P1", "xyz_class": "X"}]
    matrix, _ = abc_xyz_matrix(abc, xyz)
    assert matrix["AX"]["count"] == 1


def test_inventory_health_score_is_bounded():
    score = inventory_health_score(stockout_risk=1.0, excess_risk=1.0, coverage_days=0.0, lead_time_days=5, wmape=100, coefficient_of_variation=2.0, recommended_order=100, target_inventory=50)
    assert 0 <= score <= 100


def test_inventory_health_score_healthy():
    score = inventory_health_score(stockout_risk=0.2, excess_risk=0.1, coverage_days=5, lead_time_days=5, wmape=10, coefficient_of_variation=0.3, recommended_order=5, target_inventory=50)
    assert score >= 75


def test_inventory_health_score_critical():
    score = inventory_health_score(stockout_risk=0.9, excess_risk=0.8, coverage_days=0.1, lead_time_days=5, wmape=80, coefficient_of_variation=1.5, recommended_order=100, target_inventory=10)
    assert score <= 24


def test_inventory_health_score_zero_safe():
    score = inventory_health_score(stockout_risk=0, excess_risk=0, coverage_days=0, lead_time_days=0, wmape=0, coefficient_of_variation=0, recommended_order=0, target_inventory=0)
    assert score >= 0


def test_inventory_health_score_nan_protected():
    score = inventory_health_score(stockout_risk=float("nan"), excess_risk=float("inf"), coverage_days=float("nan"), lead_time_days=5, wmape=5, coefficient_of_variation=0.2, recommended_order=1, target_inventory=10)
    assert math.isfinite(score)


def test_inventory_health_score_deterministic():
    first = inventory_health_score(stockout_risk=0.3, excess_risk=0.2, coverage_days=5, lead_time_days=5, wmape=15, coefficient_of_variation=0.4, recommended_order=10, target_inventory=50)
    second = inventory_health_score(stockout_risk=0.3, excess_risk=0.2, coverage_days=5, lead_time_days=5, wmape=15, coefficient_of_variation=0.4, recommended_order=10, target_inventory=50)
    assert first == second


def test_health_band_for_score():
    assert inventory_health_band(97) == "Excellent"
    assert inventory_health_band(60) == "Watch"
    assert inventory_health_band(10) == "Critical"


def test_stockout_risk_low():
    result = stockout_risk_intelligence(current_inventory=100, lead_time_demand=30, reorder_point=40, safety_stock=10)
    assert result["level"] == "Low"


def test_stockout_risk_medium():
    result = stockout_risk_intelligence(current_inventory=35, lead_time_demand=30, reorder_point=40, safety_stock=10)
    assert result["level"] == "Medium"


def test_stockout_risk_high():
    result = stockout_risk_intelligence(current_inventory=20, lead_time_demand=30, reorder_point=40, safety_stock=10)
    assert result["level"] == "High"


def test_stockout_risk_critical():
    result = stockout_risk_intelligence(current_inventory=5, lead_time_demand=30, reorder_point=40, safety_stock=10)
    assert result["level"] == "Critical"


def test_stockout_risk_zero_inventory():
    result = stockout_risk_intelligence(current_inventory=0, lead_time_demand=10, reorder_point=20, safety_stock=10)
    assert result["score"] >= 90


def test_excess_inventory_normal():
    result = excess_inventory_intelligence(current_inventory=40, target_inventory=50, coverage_days=20, demand_variability=0.4, recommended_order=10)
    assert result["excess_risk_level"] in {"Low", "Medium"}


def test_excess_inventory_high():
    result = excess_inventory_intelligence(current_inventory=200, target_inventory=50, coverage_days=60, demand_variability=0.1, recommended_order=0)
    assert result["excess_risk_level"] == "High"


def test_excess_inventory_zero_demand():
    result = excess_inventory_intelligence(current_inventory=80, target_inventory=0, coverage_days=80, demand_variability=0, recommended_order=0)
    assert result["excess_inventory_percentage"] >= 0


def test_excess_inventory_zero_inventory():
    result = excess_inventory_intelligence(current_inventory=0, target_inventory=50, coverage_days=0, demand_variability=0.3, recommended_order=0)
    assert result["excess_inventory_units"] == 0


def test_opportunity_detection_reorder():
    rows = [{"product_id": "P1", "store_id": 1, "category": "Home", "region": "North", "current_inventory": 10, "lead_time_demand": 30, "reorder_point": 40, "health_score": 35, "stockout_risk_level": "Critical", "excess_risk_level": "Low", "coefficient_of_variation": 0.2, "wmape": 12, "recommended_order": 30, "target_inventory": 40}]
    opportunities = opportunity_detection(rows)
    assert any(item["opportunity_type"] in {"URGENT_REORDER", "HIGH_STOCKOUT_RISK"} for item in opportunities)


def test_opportunity_detection_stockout():
    rows = [{"product_id": "P2", "store_id": 2, "category": "Home", "region": "South", "current_inventory": 20, "lead_time_demand": 12, "reorder_point": 25, "health_score": 40, "stockout_risk_level": "High", "excess_risk_level": "Low", "coefficient_of_variation": 0.4, "wmape": 10, "recommended_order": 5, "target_inventory": 25}]
    opportunities = opportunity_detection(rows)
    assert any(item["opportunity_type"] == "HIGH_STOCKOUT_RISK" for item in opportunities)


def test_opportunity_detection_excess():
    rows = [{"product_id": "P3", "store_id": 3, "category": "Home", "region": "West", "current_inventory": 140, "lead_time_demand": 12, "reorder_point": 20, "health_score": 80, "stockout_risk_level": "Low", "excess_risk_level": "High", "excess_inventory_percentage": 80, "coefficient_of_variation": 0.3, "wmape": 10, "recommended_order": 0, "target_inventory": 50}]
    opportunities = opportunity_detection(rows)
    assert any(item["opportunity_type"] == "EXCESS_INVENTORY" for item in opportunities)


def test_opportunity_detection_accuracy():
    rows = [{"product_id": "P4", "store_id": 4, "category": "Home", "region": "East", "current_inventory": 50, "lead_time_demand": 15, "reorder_point": 20, "health_score": 60, "stockout_risk_level": "Low", "excess_risk_level": "Low", "coefficient_of_variation": 0.4, "wmape": 30, "recommended_order": 0, "target_inventory": 20}]
    opportunities = opportunity_detection(rows)
    assert any(item["opportunity_type"] == "FORECAST_ACCURACY_ISSUE" for item in opportunities)


def test_opportunity_detection_volatility():
    rows = [{"product_id": "P5", "store_id": 5, "category": "Electronics", "region": "North", "current_inventory": 80, "lead_time_demand": 20, "reorder_point": 25, "health_score": 78, "stockout_risk_level": "Low", "excess_risk_level": "Low", "coefficient_of_variation": 2.0, "wmape": 8, "recommended_order": 0, "target_inventory": 25}]
    opportunities = opportunity_detection(rows)
    assert any(item["opportunity_type"] == "HIGH_DEMAND_VOLATILITY" for item in opportunities)


def test_opportunity_ranking():
    rows = [
        {"product_id": "P1", "store_id": 1, "current_inventory": 10, "lead_time_demand": 30, "reorder_point": 40, "health_score": 20, "stockout_risk_level": "Critical", "excess_risk_level": "Low", "coefficient_of_variation": 0.2, "wmape": 5, "recommended_order": 30, "target_inventory": 40},
        {"product_id": "P2", "store_id": 2, "current_inventory": 200, "lead_time_demand": 12, "reorder_point": 20, "health_score": 70, "stockout_risk_level": "Low", "excess_risk_level": "High", "excess_inventory_percentage": 60, "coefficient_of_variation": 0.3, "wmape": 5, "recommended_order": 0, "target_inventory": 50},
    ]
    opportunities = opportunity_detection(rows)
    assert opportunities[0]["priority"] in {"Critical", "High"}


def test_service_level_increases_with_higher_service_level():
    low = service_level_analysis(demand_std=10, lead_time_days=5, current_inventory=50, service_levels=[0.90, 0.95])[0]
    high = service_level_analysis(demand_std=10, lead_time_days=5, current_inventory=50, service_levels=[0.90, 0.95])[1]
    assert high["safety_stock"] >= low["safety_stock"]


def test_inventory_intelligence_api_no_filters():
    response = client.get("/api/v1/analytics/inventory-intelligence")
    assert response.status_code == 200
    payload = response.json()
    assert {"summary", "inventory_health", "risk", "abc_xyz", "opportunities", "service_level"} <= payload.keys()


def test_inventory_intelligence_api_product_filter():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"product_id": "P100"})
    assert response.status_code == 200
    assert response.json()["summary"]["total_products"] >= 1


def test_inventory_intelligence_api_store_filter():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"store_id": 1})
    assert response.status_code == 200


def test_inventory_intelligence_api_abc_filter():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"abc_class": "A"})
    assert response.status_code in {200, 400}


def test_inventory_intelligence_api_xyz_filter():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"xyz_class": "X"})
    assert response.status_code in {200, 400}


def test_inventory_intelligence_api_risk_filter():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"risk_level": "Low"})
    assert response.status_code in {200, 400}


def test_inventory_intelligence_api_invalid_filters():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"service_level": 1.5})
    assert response.status_code == 422


def test_inventory_intelligence_api_empty_result():
    response = client.get("/api/v1/analytics/inventory-intelligence", params={"product_id": "NO_SUCH_PRODUCT"})
    assert response.status_code == 400


def test_inventory_intelligence_response_schema():
    response = client.get("/api/v1/analytics/inventory-intelligence")
    payload = response.json()
    assert isinstance(payload["summary"]["abc_distribution"], dict)
    assert isinstance(payload["service_level"], list)
    assert isinstance(payload["opportunities"], list)
