from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np


SERVICE_LEVEL_Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def service_level_z_score(service_level: float) -> float:
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1.")
    for configured_level, z_score in SERVICE_LEVEL_Z_SCORES.items():
        if abs(service_level - configured_level) < 1e-9:
            return z_score
    return NormalDist().inv_cdf(service_level)


def calculate_lead_time_demand(
    average_daily_demand: float,
    lead_time_days: float,
    forecast_demand: Sequence[float] | None = None,
) -> float:
    if average_daily_demand < 0 or lead_time_days < 0:
        raise ValueError("Demand and lead time cannot be negative.")
    daily_demand = float(np.mean(forecast_demand)) if forecast_demand else average_daily_demand
    if daily_demand < 0:
        raise ValueError("Forecast demand cannot be negative.")
    return daily_demand * lead_time_days


def calculate_demand_std(historical_demand: Sequence[float], window: int = 30) -> float:
    values = np.asarray(list(historical_demand)[-window:], dtype=float)
    if values.size == 0:
        raise ValueError("Demand history is required to calculate variability.")
    if np.any(values < 0):
        raise ValueError("Historical demand cannot be negative.")
    return max(float(np.std(values, ddof=0)), 0.0)


def calculate_safety_stock(demand_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    if demand_std < 0 or lead_time_days < 0:
        raise ValueError("Demand variability and lead time cannot be negative.")
    return max(0.0, service_level_z_score(service_level) * demand_std * sqrt(lead_time_days))


def calculate_reorder_point(lead_time_demand: float, safety_stock: float) -> float:
    if lead_time_demand < 0 or safety_stock < 0:
        raise ValueError("Reorder point inputs cannot be negative.")
    return lead_time_demand + safety_stock


def calculate_target_inventory(lead_time_demand: float, safety_stock: float) -> float:
    return calculate_reorder_point(lead_time_demand, safety_stock)


def calculate_recommended_order(target_inventory: float, current_inventory: float) -> float:
    if target_inventory < 0 or current_inventory < 0:
        raise ValueError("Inventory values cannot be negative.")
    return max(0.0, target_inventory - current_inventory)


def calculate_inventory_risk(current_inventory: float, lead_time_demand: float, reorder_point: float) -> tuple[str, float]:
    if min(current_inventory, lead_time_demand, reorder_point) < 0:
        raise ValueError("Inventory risk inputs cannot be negative.")
    if current_inventory < lead_time_demand:
        return "CRITICAL", 1.0
    if current_inventory < reorder_point:
        return "HIGH", 0.75
    if current_inventory < reorder_point * 1.25:
        return "MEDIUM", 0.5
    return "LOW", 0.0


def calculate_excess_inventory_risk(current_inventory: float, target_inventory: float) -> tuple[str, float]:
    if min(current_inventory, target_inventory) < 0:
        raise ValueError("Inventory values cannot be negative.")
    if target_inventory == 0:
        return ("HIGH", 1.0) if current_inventory > 0 else ("LOW", 0.0)
    ratio = current_inventory / target_inventory
    if ratio >= 1.5:
        return "HIGH", min(1.0, (ratio - 1.5) / 1.5 + 0.5)
    if ratio > 1.25:
        return "MEDIUM", 0.5
    return "LOW", 0.0


def inventory_recommendation(df, forecast_points, service_level: float = 0.95) -> dict[str, float | str | None]:
    if df.empty:
        raise ValueError("Demand history is required for inventory optimization.")
    required = {"units_sold", "inventory_on_hand", "supplier_lead_time_days"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing inventory fields: {', '.join(sorted(missing))}.")

    current_inventory = float(df["inventory_on_hand"].iloc[-1])
    lead_time = float(df["supplier_lead_time_days"].iloc[-1])
    if current_inventory < 0 or lead_time < 0:
        raise ValueError("Current inventory and lead time cannot be negative.")
    history = df["units_sold"].tail(30).astype(float).tolist()
    average_daily_demand = float(np.mean(history)) if history else 0.0
    forecast_values = [float(point["forecast_demand"]) for point in forecast_points]
    lead_time_demand = calculate_lead_time_demand(average_daily_demand, lead_time, forecast_values)
    demand_std = calculate_demand_std(history)
    safety_stock = calculate_safety_stock(demand_std, lead_time, service_level)
    reorder_point = calculate_reorder_point(lead_time_demand, safety_stock)
    target_inventory = calculate_target_inventory(lead_time_demand, safety_stock)
    stockout_label, stockout_risk = calculate_inventory_risk(current_inventory, lead_time_demand, reorder_point)
    excess_label, excess_risk = calculate_excess_inventory_risk(current_inventory, target_inventory)
    coverage = current_inventory / average_daily_demand if average_daily_demand > 0 else None
    forecast_total = float(sum(forecast_values))
    average_forecast_error = float(np.mean([abs(float(point["upper_bound"]) - float(point["forecast_demand"])) for point in forecast_points])) if forecast_points else 0.0

    return {
        "forecast_total": forecast_total,
        "average_daily_demand": average_daily_demand,
        "lead_time_days": lead_time,
        "lead_time_demand": lead_time_demand,
        "demand_std": demand_std,
        "service_level": service_level,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "target_inventory": target_inventory,
        "recommended_order": calculate_recommended_order(target_inventory, current_inventory),
        "inventory_coverage_days": coverage,
        "stockout_risk": stockout_risk,
        "stockout_label": stockout_label,
        "excess_inventory_risk": excess_risk,
        "excess_label": excess_label,
        "inventory_risk": max(stockout_risk, excess_risk),
        "average_forecast_error": average_forecast_error,
    }
