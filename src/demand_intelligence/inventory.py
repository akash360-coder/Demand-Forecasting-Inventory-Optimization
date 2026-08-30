from __future__ import annotations

from math import sqrt

import numpy as np


def inventory_recommendation(df, forecast_points, service_level: float = 0.95) -> dict[str, float]:
    if not forecast_points:
        return {
            "forecast_total": 0.0,
            "recommended_order": 0.0,
            "stockout_risk": 0.0,
            "excess_inventory_risk": 0.0,
            "inventory_risk": 0.0,
            "average_forecast_error": 0.0,
        }

    forecast_total = float(sum(point["forecast_demand"] for point in forecast_points))
    current_inventory = float(df["inventory_on_hand"].iloc[-1])
    lead_time = float(df["supplier_lead_time_days"].iloc[-1])
    demand_std = max(float(np.std(df["units_sold"].tail(30))), 1.0)
    z_score = 1.645 if service_level >= 0.95 else 1.28
    safety_stock = z_score * demand_std * sqrt(max(lead_time, 1))
    recommended_order = max(0.0, forecast_total + safety_stock - current_inventory)
    stockout_risk = min(1.0, max(0.0, (safety_stock - current_inventory + forecast_total) / (forecast_total + demand_std + 1e-6)))
    excess_inventory_risk = min(1.0, max(0.0, (current_inventory - forecast_total) / (forecast_total + current_inventory + 1e-6)))
    inventory_risk = max(stockout_risk, excess_inventory_risk)
    average_forecast_error = float(np.mean([abs(point["upper_bound"] - point["forecast_demand"]) for point in forecast_points]))

    return {
        "forecast_total": forecast_total,
        "recommended_order": recommended_order,
        "stockout_risk": stockout_risk,
        "excess_inventory_risk": excess_inventory_risk,
        "inventory_risk": inventory_risk,
        "average_forecast_error": average_forecast_error,
    }
