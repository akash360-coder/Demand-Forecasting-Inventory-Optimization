from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.demand_intelligence.inventory import (
    calculate_demand_std,
    calculate_excess_inventory_risk,
    calculate_inventory_risk,
    calculate_recommended_order,
    calculate_reorder_point,
    calculate_safety_stock,
    service_level_z_score,
)

ABC_A_THRESHOLD = 0.80
ABC_B_THRESHOLD = 0.95
XYZ_X_THRESHOLD = 0.15
XYZ_Y_THRESHOLD = 0.50
HEALTH_BANDS = (
    (90, 100, "Excellent"),
    (75, 89, "Healthy"),
    (50, 74, "Watch"),
    (25, 49, "Risk"),
    (0, 24, "Critical"),
)
RISK_LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(parsed) or math.isinf(parsed):
        return float(default)
    return float(parsed)


def _as_series(values: Iterable[float]) -> pd.Series:
    return pd.Series([_safe_float(item) for item in values], dtype=float)


def _percentage_share(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, part / total))


def _classify_value(value: float, a_threshold: float, b_threshold: float) -> str:
    epsilon = 1e-9
    if value <= a_threshold + epsilon:
        return "A"
    if value <= b_threshold + epsilon:
        return "B"
    return "C"


def _classify_xyz_value(cv: float, x_threshold: float, y_threshold: float) -> str:
    if cv <= 0 or (not np.isfinite(cv)):
        return "X"
    if cv <= x_threshold + 1e-9:
        return "X"
    if cv <= y_threshold + 1e-9:
        return "Y"
    return "Z"


def classify_abc(
    frame: pd.DataFrame,
    *,
    product_id_col: str = "product_id",
    value_col: str = "business_value",
    a_threshold: float = ABC_A_THRESHOLD,
    b_threshold: float = ABC_B_THRESHOLD,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if not {product_id_col, value_col}.issubset(frame.columns):
        raise ValueError(f"ABC analysis requires columns {product_id_col!r} and {value_col!r}.")
    safe_frame = frame[[product_id_col, value_col]].copy()
    safe_frame[value_col] = pd.to_numeric(safe_frame[value_col], errors="coerce").fillna(0.0)
    safe_frame = safe_frame[safe_frame[product_id_col].notna()].copy()
    if safe_frame.empty:
        return []
    total_value = float(safe_frame[value_col].sum())
    ranked = safe_frame.assign(
        business_value=safe_frame[value_col].astype(float),
        product_id=safe_frame[product_id_col].astype(str),
    ).sort_values(["business_value", "product_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    if total_value <= 0:
        return [
            {
                "product_id": str(row[product_id_col]),
                "business_value": float(row[value_col]),
                "percentage_of_total": 0.0,
                "cumulative_percentage": 0.0,
                "abc_class": "C",
            }
            for _, row in ranked.iterrows()
        ]

    cumulative = 0.0
    results: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        business_value = float(row[value_col])
        percentage = _percentage_share(business_value, total_value)
        cumulative += percentage
        abc_class = _classify_value(cumulative, a_threshold, b_threshold)
        results.append(
            {
                "product_id": str(row[product_id_col]),
                "business_value": business_value,
                "percentage_of_total": percentage,
                "cumulative_percentage": cumulative,
                "abc_class": abc_class,
            }
        )
    return results


def classify_xyz(
    frame: pd.DataFrame,
    *,
    product_id_col: str = "product_id",
    demand_col: str = "units_sold",
    x_threshold: float = XYZ_X_THRESHOLD,
    y_threshold: float = XYZ_Y_THRESHOLD,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if not {product_id_col, demand_col}.issubset(frame.columns):
        raise ValueError(f"XYZ analysis requires columns {product_id_col!r} and {demand_col!r}.")
    safe_frame = frame[[product_id_col, demand_col]].copy()
    safe_frame[demand_col] = pd.to_numeric(safe_frame[demand_col], errors="coerce").fillna(0.0)
    safe_frame = safe_frame[safe_frame[product_id_col].notna()].copy()
    if safe_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for product_id, group in safe_frame.groupby(product_id_col, sort=True):
        values = group[demand_col].astype(float).to_numpy()
        mean_demand = float(np.mean(values)) if values.size else 0.0
        demand_std = float(np.std(values, ddof=0)) if values.size else 0.0
        if mean_demand > 0:
            coefficient_of_variation = demand_std / mean_demand
        else:
            coefficient_of_variation = 0.0
        xyz_class = _classify_xyz_value(coefficient_of_variation, x_threshold, y_threshold)
        rows.append(
            {
                "product_id": str(product_id),
                "mean_demand": max(0.0, mean_demand),
                "demand_std": max(0.0, demand_std),
                "coefficient_of_variation": max(0.0, coefficient_of_variation),
                "xyz_class": xyz_class,
            }
        )
    return rows


def abc_xyz_matrix(
    abc_rows: list[dict[str, Any]],
    xyz_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | int]], dict[str, str]]:
    mapping = {item["product_id"]: item for item in xyz_rows}
    matrix = {"AX": {"count": 0, "business_value": 0.0}, "AY": {"count": 0, "business_value": 0.0}, "AZ": {"count": 0, "business_value": 0.0}, "BX": {"count": 0, "business_value": 0.0}, "BY": {"count": 0, "business_value": 0.0}, "BZ": {"count": 0, "business_value": 0.0}, "CX": {"count": 0, "business_value": 0.0}, "CY": {"count": 0, "business_value": 0.0}, "CZ": {"count": 0, "business_value": 0.0}}
    interpretations = {
        "AX": "High-value and highly predictable.",
        "AY": "High-value with moderate variability; monitor demand shifts.",
        "AZ": "High-value but highly unpredictable; use stronger scenario analysis.",
        "BX": "Moderate value and predictable; maintain steady replenishment discipline.",
        "BY": "Moderate value with moderate variability; watch replenishment stability.",
        "BZ": "Moderate value and volatile; maintain tighter monitoring.",
        "CX": "Lower-value and predictable; simpler replenishment strategy may be appropriate.",
        "CY": "Lower-value but moderate variability; review operational complexity.",
        "CZ": "Lower-value and highly unpredictable; review whether complexity is justified.",
    }
    for item in abc_rows:
        product_id = str(item["product_id"])
        xyz_item = mapping.get(product_id)
        if xyz_item is None:
            continue
        segment = f"{item['abc_class']}{xyz_item['xyz_class']}"
        matrix.setdefault(segment, {"count": 0, "business_value": 0.0})
        matrix[segment]["count"] = int(matrix[segment]["count"]) + 1
        matrix[segment]["business_value"] = float(matrix[segment]["business_value"]) + float(item["business_value"])
    return matrix, interpretations


def inventory_health_score(
    *,
    stockout_risk: float | int = 0.0,
    excess_risk: float | int = 0.0,
    coverage_days: float | int | None = None,
    lead_time_days: float | int = 0.0,
    wmape: float | int = 0.0,
    coefficient_of_variation: float | int = 0.0,
    recommended_order: float | int = 0.0,
    target_inventory: float | int = 0.0,
) -> float:
    stockout_risk = _safe_float(stockout_risk)
    excess_risk = _safe_float(excess_risk)
    coverage_days = _safe_float(coverage_days, default=0.0) if coverage_days is not None else 0.0
    lead_time_days = _safe_float(lead_time_days)
    wmape = _safe_float(wmape)
    coefficient_of_variation = _safe_float(coefficient_of_variation)
    recommended_order = _safe_float(recommended_order)
    target_inventory = _safe_float(target_inventory)

    stockout_component = 100.0 * (1.0 - min(1.0, stockout_risk))
    excess_component = 100.0 * (1.0 - min(1.0, excess_risk))
    accuracy_component = 100.0 * max(0.0, 1.0 - min(1.0, wmape / 100.0))
    volatility_component = 100.0 * max(0.0, 1.0 - min(1.0, coefficient_of_variation))
    if lead_time_days > 0:
        coverage_component = 100.0 * min(1.0, max(0.0, coverage_days / lead_time_days))
    else:
        coverage_component = 100.0
    if target_inventory > 0:
        order_component = 100.0 * max(0.0, 1.0 - min(1.0, recommended_order / target_inventory))
    else:
        order_component = 100.0
    score = (
        0.30 * stockout_component
        + 0.20 * excess_component
        + 0.20 * coverage_component
        + 0.15 * accuracy_component
        + 0.15 * volatility_component
        + 0.10 * order_component
    )
    return float(np.clip(score, 0.0, 100.0))


def inventory_health_band(score: float) -> str:
    safe_score = _safe_float(score, default=0.0)
    for low, high, label in HEALTH_BANDS:
        if low <= safe_score <= high:
            return label
    return "Critical"


def stockout_risk_intelligence(
    *,
    current_inventory: float,
    lead_time_demand: float,
    reorder_point: float,
    safety_stock: float,
    forecast_demand: float | None = None,
) -> dict[str, Any]:
    current_inventory = _safe_float(current_inventory)
    lead_time_demand = _safe_float(lead_time_demand)
    reorder_point = _safe_float(reorder_point)
    safety_stock = _safe_float(safety_stock)
    if reorder_point <= 0:
        return {"score": 0.0, "level": "Low", "projected_inventory_pressure": 0.0, "recommended_operational_attention": "No immediate inventory risk signal."}
    if current_inventory <= 0:
        level = "Critical"
        score = 100.0
    elif current_inventory <= max(0.25 * reorder_point, 0.5 * lead_time_demand):
        level = "Critical"
        score = min(100.0, 100.0 * (1.0 - (current_inventory / max(reorder_point, 1.0))))
    elif current_inventory < lead_time_demand or current_inventory <= 0.75 * reorder_point:
        level = "High"
        score = min(100.0, 100.0 * (1.0 - (current_inventory / max(reorder_point, 1.0))))
    elif current_inventory < reorder_point:
        level = "Medium"
        score = min(100.0, 75.0 * (1.0 - (current_inventory / max(reorder_point, 1.0))))
    elif current_inventory < reorder_point * 1.25:
        level = "Medium"
        score = min(100.0, 60.0 * (1.0 - (current_inventory / max(reorder_point * 1.25, 1.0))))
    else:
        level = "Low"
        score = 0.0
    projected_pressure = max(0.0, reorder_point - current_inventory)
    if projected_pressure > 0:
        attention = "Increase monitoring around replenishment timing and service coverage."
    elif current_inventory <= 0:
        attention = "Immediate replenishment review is recommended."
    else:
        attention = "Inventory remains within an acceptable replenishment buffer."
    if level == "Critical":
        attention = "Critical stockout risk: prioritize replenishment review and verify supplier capacity."
    elif level == "High":
        attention = "High stockout risk: review purchase timing and safety stock coverage."
    elif level == "Medium":
        attention = "Medium stockout risk: monitor ordering cadence and volatility."
    return {
        "score": float(np.clip(score, 0.0, 100.0)),
        "level": level,
        "projected_inventory_pressure": float(projected_pressure),
        "recommended_operational_attention": attention,
    }


def excess_inventory_intelligence(
    *,
    current_inventory: float,
    target_inventory: float,
    coverage_days: float | None = None,
    demand_variability: float | None = None,
    recommended_order: float | None = None,
) -> dict[str, Any]:
    current_inventory = _safe_float(current_inventory)
    target_inventory = _safe_float(target_inventory)
    coverage_days = _safe_float(coverage_days, default=0.0) if coverage_days is not None else 0.0
    demand_variability = _safe_float(demand_variability, default=0.0) if demand_variability is not None else 0.0
    recommended_order = _safe_float(recommended_order, default=0.0) if recommended_order is not None else 0.0
    excess_units = max(0.0, current_inventory - target_inventory)
    if target_inventory > 0:
        excess_percentage = (excess_units / target_inventory) * 100.0
    else:
        excess_percentage = 0.0 if current_inventory <= 0 else 100.0
    if current_inventory <= 0:
        level = "Low"
        reason = "Inventory is at or below target; no meaningful excess inventory signal detected."
    elif excess_units <= 0:
        level = "Low"
        reason = "Current inventory remains at or below the target buffer."
    elif excess_percentage >= 50 or (coverage_days > 0 and coverage_days > 1.5 * max(30.0, demand_variability * 30.0)):
        level = "High"
        reason = "Inventory exceeds the target buffer and coverage remains materially above operational need."
    elif excess_percentage >= 20:
        level = "Medium"
        reason = "Inventory coverage is above the target, suggesting slower movement or elevated buffer."
    else:
        level = "Low"
        reason = "Inventory is modestly above target, but within a manageable operational range."
    if recommended_order > 0 and excess_units > 0:
        reason = f"{reason} Recommended buy quantity is {recommended_order:.2f} units against a target of {target_inventory:.2f} units."
    return {
        "excess_inventory_units": float(excess_units),
        "excess_inventory_percentage": float(np.clip(excess_percentage, 0.0, 100.0)),
        "excess_risk_level": level,
        "inventory_coverage": float(coverage_days),
        "reason": reason,
    }


def opportunity_detection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    for row in rows:
        product_id = str(row.get("product_id", "UNKNOWN"))
        store_id = row.get("store_id", 0)
        category = row.get("category")
        region = row.get("region")
        health_score = _safe_float(row.get("health_score"), 0.0)
        stockout_level = row.get("stockout_risk_level", "Low")
        stockout_score = _safe_float(row.get("stockout_risk_score"), 0.0)
        excess_level = row.get("excess_risk_level", "Low")
        excess_percentage = _safe_float(row.get("excess_inventory_percentage"), 0.0)
        cv = _safe_float(row.get("coefficient_of_variation"), 0.0)
        wmape = _safe_float(row.get("wmape"), 0.0)
        recommended_order = _safe_float(row.get("recommended_order"), 0.0)
        threshold = _safe_float(row.get("reorder_point"), 0.0)

        if stockout_level in {"High", "Critical"}:
            opportunity_type = "URGENT_REORDER" if row.get("current_inventory", 0) < row.get("lead_time_demand", 0) else "HIGH_STOCKOUT_RISK"
            metric = "current_inventory"
            current_value = _safe_float(row.get("current_inventory"), 0.0)
            priority = "Critical" if stockout_level == "Critical" else "High"
            explanation = f"Current inventory {current_value:.2f} is below the operational trigger {threshold:.2f}."
            opportunities.append({
                "product_id": product_id,
                "store_id": store_id,
                "category": category,
                "region": region,
                "priority": priority,
                "opportunity_type": opportunity_type,
                "relevant_metric": metric,
                "current_value": current_value,
                "threshold": threshold,
                "explanation": explanation,
            })
        if excess_level in {"High", "Medium"}:
            current_value = _safe_float(row.get("current_inventory"), 0.0)
            threshold = _safe_float(row.get("target_inventory"), 0.0)
            opportunities.append({
                "product_id": product_id,
                "store_id": store_id,
                "category": category,
                "region": region,
                "priority": "Medium" if excess_level == "Medium" else "High",
                "opportunity_type": "EXCESS_INVENTORY",
                "relevant_metric": "excess_inventory_percentage",
                "current_value": current_value,
                "threshold": threshold,
                "explanation": f"Current inventory exceeds the target by {excess_percentage:.1f}% and coverage is above the operational target.",
            })
        if wmape > 25:
            opportunities.append({
                "product_id": product_id,
                "store_id": store_id,
                "category": category,
                "region": region,
                "priority": "Medium",
                "opportunity_type": "FORECAST_ACCURACY_ISSUE",
                "relevant_metric": "wmape",
                "current_value": wmape,
                "threshold": 25.0,
                "explanation": f"Forecast accuracy is elevated at {wmape:.1f}% WMAPE, suggesting a need to review model and replenishment assumptions.",
            })
        if cv > 1.0:
            opportunities.append({
                "product_id": product_id,
                "store_id": store_id,
                "category": category,
                "region": region,
                "priority": "Medium",
                "opportunity_type": "HIGH_DEMAND_VOLATILITY",
                "relevant_metric": "coefficient_of_variation",
                "current_value": cv,
                "threshold": 1.0,
                "explanation": f"Demand variability is high (CV={cv:.2f}), which increases replenishment and service volatility.",
            })
        if health_score < 50 and not any(item["opportunity_type"] in {"URGENT_REORDER", "HIGH_STOCKOUT_RISK", "EXCESS_INVENTORY"} for item in opportunities if item["product_id"] == product_id and item["store_id"] == store_id):
            opportunities.append({
                "product_id": product_id,
                "store_id": store_id,
                "category": category,
                "region": region,
                "priority": "Low",
                "opportunity_type": "HEALTHY_INVENTORY",
                "relevant_metric": "inventory_health_score",
                "current_value": health_score,
                "threshold": 50.0,
                "explanation": "Inventory health score is below the watch threshold, so keep a close operational review on replenishment conditions.",
            })
    priority_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    return sorted(opportunities, key=lambda item: (-priority_order.get(str(item["priority"]), 0), str(item["product_id"]), str(item.get("store_id", ""))))


def product_store_risk_matrix(
    rows: list[dict[str, Any]],
    *,
    product_id: str | None = None,
    store_id: int | None = None,
    category: str | None = None,
    region: str | None = None,
    abc_class: str | None = None,
    xyz_class: str | None = None,
    risk_level: str | None = None,
) -> dict[str, Any]:
    filtered = []
    for row in rows:
        if product_id and str(row.get("product_id")) != str(product_id):
            continue
        if store_id is not None and int(row.get("store_id", -1)) != int(store_id):
            continue
        if category and str(row.get("category", "")).lower() != str(category).lower():
            continue
        if region and str(row.get("region", "")).lower() != str(region).lower():
            continue
        if abc_class and str(row.get("abc_class", "")).upper() != str(abc_class).upper():
            continue
        if xyz_class and str(row.get("xyz_class", "")).upper() != str(xyz_class).upper():
            continue
        if risk_level and str(row.get("stockout_risk_level", "")).upper() != str(risk_level).upper():
            continue
        filtered.append(row)
    aggregate = {
        "total_records": len(filtered),
        "healthy": sum(1 for row in filtered if row.get("health_band") == "Excellent" or row.get("health_band") == "Healthy"),
        "watch": sum(1 for row in filtered if row.get("health_band") == "Watch"),
        "risk": sum(1 for row in filtered if row.get("health_band") in {"Risk", "Critical"}),
        "high_stockout": sum(1 for row in filtered if row.get("stockout_risk_level") in {"High", "Critical"}),
        "excess_inventory": sum(1 for row in filtered if row.get("excess_risk_level") in {"Medium", "High"}),
    }
    details = [
        {
            "product_id": row.get("product_id"),
            "store_id": row.get("store_id"),
            "category": row.get("category"),
            "region": row.get("region"),
            "abc_class": row.get("abc_class"),
            "xyz_class": row.get("xyz_class"),
            "health_score": _safe_float(row.get("health_score"), 0.0),
            "stockout_risk_score": _safe_float(row.get("stockout_risk_score"), 0.0),
            "stockout_risk_level": row.get("stockout_risk_level", "Low"),
            "excess_inventory_percentage": _safe_float(row.get("excess_inventory_percentage"), 0.0),
            "health_band": row.get("health_band", "Critical"),
        }
        for row in filtered
    ]
    return {"aggregate": aggregate, "details": details}


def service_level_analysis(
    *,
    demand_std: float,
    lead_time_days: float,
    current_inventory: float,
    service_levels: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    service_levels = list(service_levels or [0.90, 0.95, 0.98, 0.99])
    lead_time_demand = max(0.0, (demand_std if demand_std > 0 else 0.0) * lead_time_days)
    rows: list[dict[str, Any]] = []
    for service_level in service_levels:
        level = _safe_float(service_level)
        z_score = service_level_z_score(level)
        safety_stock = calculate_safety_stock(demand_std, lead_time_days, level)
        reorder_point = calculate_reorder_point(lead_time_demand, safety_stock)
        target_inventory = reorder_point
        recommended_order = calculate_recommended_order(target_inventory, current_inventory)
        rows.append(
            {
                "service_level": float(level),
                "z_score": float(z_score),
                "safety_stock": float(safety_stock),
                "reorder_point": float(reorder_point),
                "target_inventory": float(target_inventory),
                "recommended_order": float(recommended_order),
            }
        )
    return rows


def build_inventory_intelligence_dataframe(frame: pd.DataFrame, *, service_level: float = 0.95) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["product_id", "store_id", "category", "region", "business_value", "mean_demand", "demand_std", "coefficient_of_variation", "current_inventory", "lead_time_days", "lead_time_demand", "reorder_point", "target_inventory", "recommended_order", "stockout_risk_score", "stockout_risk_level", "excess_inventory_units", "excess_inventory_percentage", "excess_risk_level", "inventory_coverage", "health_score", "health_band", "wmape", "abc_class", "xyz_class"])
    safe_frame = frame.copy()
    safe_frame["date"] = pd.to_datetime(safe_frame["date"], errors="coerce")
    safe_frame["units_sold"] = pd.to_numeric(safe_frame["units_sold"], errors="coerce").fillna(0.0)
    safe_frame["price"] = pd.to_numeric(safe_frame["price"], errors="coerce").fillna(0.0)
    safe_frame["inventory_on_hand"] = pd.to_numeric(safe_frame["inventory_on_hand"], errors="coerce").fillna(0.0)
    safe_frame["supplier_lead_time_days"] = pd.to_numeric(safe_frame["supplier_lead_time_days"], errors="coerce").fillna(0.0)
    safe_frame = safe_frame[safe_frame["product_id"].notna()].copy()
    rows = []
    for (product_id, store_id), group in safe_frame.groupby(["product_id", "store_id"], dropna=False, sort=True):
        values = group["units_sold"].astype(float)
        value_total = float((values * group["price"]).sum()) if "price" in group.columns else 0.0
        mean_demand = float(values.mean()) if len(values) else 0.0
        demand_std = calculate_demand_std(values.tolist()) if len(values) else 0.0
        coefficient_of_variation = (demand_std / mean_demand) if mean_demand > 0 else 0.0
        current_inventory = float(group["inventory_on_hand"].iloc[-1]) if len(group) else 0.0
        lead_time_days = float(group["supplier_lead_time_days"].iloc[-1]) if len(group) else 0.0
        lead_time_demand = max(0.0, mean_demand * lead_time_days)
        safety_stock = calculate_safety_stock(demand_std, lead_time_days, service_level)
        reorder_point = calculate_reorder_point(lead_time_demand, safety_stock)
        target_inventory = reorder_point
        recommended_order = calculate_recommended_order(target_inventory, current_inventory)
        stockout_label, stockout_score = calculate_inventory_risk(current_inventory, lead_time_demand, reorder_point)
        risk_details = stockout_risk_intelligence(
            current_inventory=current_inventory,
            lead_time_demand=lead_time_demand,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
        )
        excess_label, excess_risk = calculate_excess_inventory_risk(current_inventory, target_inventory)
        excess_details = excess_inventory_intelligence(
            current_inventory=current_inventory,
            target_inventory=target_inventory,
            coverage_days=(current_inventory / mean_demand) if mean_demand > 0 else 0.0,
            demand_variability=coefficient_of_variation,
            recommended_order=recommended_order,
        )
        coverage_days = (current_inventory / mean_demand) if mean_demand > 0 else 0.0
        health_score = inventory_health_score(
            stockout_risk=stockout_score,
            excess_risk=excess_risk,
            coverage_days=coverage_days,
            lead_time_days=lead_time_days,
            wmape=0.0,
            coefficient_of_variation=coefficient_of_variation,
            recommended_order=recommended_order,
            target_inventory=target_inventory,
        )
        rows.append(
            {
                "product_id": str(product_id),
                "store_id": int(store_id),
                "category": str(group["category"].iloc[0]) if "category" in group.columns and not group["category"].empty else "Unknown",
                "region": str(group["region"].iloc[0]) if "region" in group.columns and not group["region"].empty else "Unknown",
                "business_value": value_total,
                "mean_demand": max(0.0, mean_demand),
                "demand_std": max(0.0, demand_std),
                "coefficient_of_variation": max(0.0, coefficient_of_variation),
                "current_inventory": max(0.0, current_inventory),
                "lead_time_days": max(0.0, lead_time_days),
                "lead_time_demand": max(0.0, lead_time_demand),
                "reorder_point": max(0.0, reorder_point),
                "target_inventory": max(0.0, target_inventory),
                "recommended_order": max(0.0, recommended_order),
                "stockout_risk_score": float(stockout_score),
                "stockout_risk_level": str(stockout_label),
                "excess_inventory_units": float(excess_details["excess_inventory_units"]),
                "excess_inventory_percentage": float(excess_details["excess_inventory_percentage"]),
                "excess_risk_level": str(excess_label),
                "inventory_coverage": float(coverage_days),
                "health_score": float(health_score),
                "health_band": inventory_health_band(health_score),
                "wmape": 0.0,
            }
        )
    return pd.DataFrame(rows)
