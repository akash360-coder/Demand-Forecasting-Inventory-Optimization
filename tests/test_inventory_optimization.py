import pandas as pd
import pytest

from src.demand_intelligence.inventory import (
    calculate_demand_std,
    calculate_lead_time_demand,
    calculate_recommended_order,
    calculate_reorder_point,
    calculate_safety_stock,
    inventory_recommendation,
)


def test_lead_time_demand_and_reorder_point():
    assert calculate_lead_time_demand(25, 5) == 125
    assert calculate_reorder_point(125, 30) == 155


def test_safety_stock_is_non_negative_and_deterministic():
    assert calculate_safety_stock(10, 4, 0.95) == pytest.approx(32.898)
    assert calculate_demand_std([10, 10, 10]) == 0


def test_recommended_order_never_goes_negative():
    assert calculate_recommended_order(155, 80) == 75
    assert calculate_recommended_order(155, 200) == 0


def test_inventory_recommendation_handles_zero_demand_and_lead_time():
    frame = pd.DataFrame(
        {
            "units_sold": [0, 0, 0],
            "inventory_on_hand": [20, 20, 20],
            "supplier_lead_time_days": [0, 0, 0],
        }
    )
    result = inventory_recommendation(frame, [])
    assert result["lead_time_demand"] == 0
    assert result["safety_stock"] == 0
    assert result["inventory_coverage_days"] is None
    assert result["recommended_order"] == 0


def test_negative_inventory_is_rejected():
    frame = pd.DataFrame(
        {
            "units_sold": [10],
            "inventory_on_hand": [-1],
            "supplier_lead_time_days": [2],
        }
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        inventory_recommendation(frame, [])
