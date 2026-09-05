import os

os.environ.setdefault("DATABASE_MODE", "sqlite")

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.forecast import SimulationRequest
from app.services.simulation_service import run_simulation
from src.demand_intelligence.forecasting import load_production_model


client = TestClient(app)
VALID = {
    "product_id": "P100",
    "store_id": 1,
    "forecast_horizon": 7,
    "price": 100,
    "promotion": True,
    "holiday": False,
    "lead_time_days": 5,
    "current_inventory": 120,
}


def test_valid_simulation_request():
    response = client.post("/api/v1/simulate", json=VALID)
    assert response.status_code == 200
    assert {"baseline", "scenario", "forecast", "inventory", "impact", "explanation", "metadata"} <= response.json().keys()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("product_id", "missing"),
        ("store_id", 999),
        ("forecast_horizon", 0),
        ("forecast_horizon", 31),
        ("price", 0),
        ("price", -1),
        ("current_inventory", -1),
        ("lead_time_days", 0),
        ("lead_time_days", -1),
    ],
)
def test_invalid_simulation_input(field, value):
    payload = {**VALID, field: value}
    assert client.post("/api/v1/simulate", json=payload).status_code in {400, 422}


def test_options_return_real_entities():
    response = client.get("/api/v1/simulation/options")
    assert response.status_code == 200
    assert response.json()["products"]
    assert response.json()["stores"]


def test_persisted_artifact_is_supported_tree_model():
    artifact = load_production_model()
    assert type(artifact["model"]).__name__ in {"LGBMRegressor", "XGBRegressor", "RandomForestRegressor"}


def test_horizon_one_returns_one_finite_nonnegative_prediction():
    result = run_simulation(SimulationRequest(**{**VALID, "forecast_horizon": 1}))
    assert len(result.forecast) == 1
    assert result.forecast[0].predicted_demand >= 0


def test_horizon_seven_is_recursive_and_complete():
    result = run_simulation(SimulationRequest(**VALID))
    assert len(result.forecast) == 7
    assert all(point.predicted_demand >= 0 for point in result.forecast)


def test_inventory_formulas_have_safe_outputs():
    result = run_simulation(SimulationRequest(**VALID))
    inventory = result.inventory
    assert inventory.lead_time_demand >= 0
    assert inventory.safety_stock >= 0
    assert inventory.reorder_point == pytest.approx(inventory.lead_time_demand + inventory.safety_stock)
    assert inventory.target_inventory == pytest.approx(inventory.reorder_point)
    assert inventory.recommended_order >= 0
    assert inventory.coverage_days is None or inventory.coverage_days >= 0


def test_inventory_risk_labels_are_returned():
    inventory = run_simulation(SimulationRequest(**VALID)).inventory
    assert inventory.stockout_label in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert inventory.excess_inventory_label in {"LOW", "MEDIUM", "HIGH"}


def test_inventory_change_affects_recommended_order_not_history():
    low = run_simulation(SimulationRequest(**{**VALID, "current_inventory": 50}))
    high = run_simulation(SimulationRequest(**{**VALID, "current_inventory": 200}))
    assert low.result.inventory.recommended_order >= high.result.inventory.recommended_order


def test_lead_time_change_affects_inventory_metrics():
    short = run_simulation(SimulationRequest(**{**VALID, "lead_time_days": 3}))
    long = run_simulation(SimulationRequest(**{**VALID, "lead_time_days": 7}))
    assert long.result.inventory.lead_time_demand >= short.result.inventory.lead_time_demand


def test_promotion_scenario_is_separate_from_baseline():
    result = run_simulation(SimulationRequest(**{**VALID, "promotion": True}))
    assert result.scenario.promotion is True
    assert result.baseline.forecast != result.result.forecast


def test_price_scenario_is_applied_to_request():
    result = run_simulation(SimulationRequest(**{**VALID, "price": 250}))
    assert result.scenario.price == 250


def test_impact_matches_forecast_totals():
    result = run_simulation(SimulationRequest(**VALID))
    assert result.impact.demand_change == pytest.approx(result.result.forecast_demand - result.baseline.forecast_demand)


def test_impact_percentage_matches_demand_change():
    result = run_simulation(SimulationRequest(**VALID))
    expected = result.impact.demand_change / result.baseline.forecast_demand * 100
    assert result.impact.demand_change_percent == pytest.approx(expected)


def test_real_shap_features_are_aligned_and_ranked():
    explanation = run_simulation(SimulationRequest(**VALID)).explanation
    assert explanation.features
    assert all(item.feature and item.direction in {"positive", "negative"} for item in explanation.features)
    assert all(item.importance == pytest.approx(abs(item.shap_value)) for item in explanation.features)


def test_shap_additivity_check_passes():
    assert run_simulation(SimulationRequest(**VALID)).explanation.sanity_check is True


def test_metadata_contains_model_identity_without_path():
    metadata = run_simulation(SimulationRequest(**VALID)).metadata
    assert metadata.model_name
    assert metadata.model_version
    assert ":\\" not in metadata.model_name


def test_api_rejects_unknown_product():
    response = client.post("/api/v1/simulate", json={**VALID, "product_id": "UNKNOWN"})
    assert response.status_code == 400


def test_api_rejects_unknown_store():
    response = client.post("/api/v1/simulate", json={**VALID, "store_id": 999})
    assert response.status_code == 400
