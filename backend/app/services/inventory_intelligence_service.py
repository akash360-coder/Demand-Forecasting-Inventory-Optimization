from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.forecast import (
    InventoryABCXYZSummary,
    InventoryIntelligenceResponse,
    InventoryIntelligenceSummary,
    InventoryHealthSummary,
    InventoryOpportunity,
    InventoryRiskSummary,
    InventoryServiceLevelComparison,
)
from app.services.forecast_service import _load_dataframe
from src.demand_intelligence.inventory_intelligence import (
    abc_xyz_matrix,
    build_inventory_intelligence_dataframe,
    classify_abc,
    classify_xyz,
    excess_inventory_intelligence,
    inventory_health_band,
    inventory_health_score,
    opportunity_detection,
    product_store_risk_matrix,
    service_level_analysis,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(parsed) or np.isinf(parsed):
        return float(default)
    return float(parsed)


def _distribution_from_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "Unknown").upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _all_types_for(rows: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for row in rows:
        key = str(row.get("abc_class") or "Unknown").upper()
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _empty_inventory_response() -> InventoryIntelligenceResponse:
    return InventoryIntelligenceResponse(
        summary=InventoryIntelligenceSummary(
            total_products=0,
            total_stores=0,
            total_inventory_units=0.0,
            stockout_risk_count=0,
            excess_inventory_count=0,
            critical_inventory_count=0,
            average_health_score=0.0,
            abc_distribution={},
            xyz_distribution={},
            abc_xyz_distribution={},
        ),
        inventory_health=InventoryHealthSummary(
            average_score=0.0,
            health_band_counts={"Excellent": 0, "Healthy": 0, "Watch": 0, "Risk": 0, "Critical": 0},
            top_critical_products=[],
        ),
        risk=InventoryRiskSummary(
            stockout_risk_distribution={},
            excess_inventory_distribution={},
            risk_matrix_data={"aggregate": {"total_records": 0, "healthy": 0, "watch": 0, "risk": 0, "high_stockout": 0, "excess_inventory": 0}, "details": []},
        ),
        abc_xyz=[],
        opportunities=[],
        service_level=[],
    )


def get_inventory_intelligence_response(
    product_id: str | None = None,
    store_id: int | None = None,
    category: str | None = None,
    region: str | None = None,
    abc_class: str | None = None,
    xyz_class: str | None = None,
    risk_level: str | None = None,
    health_band: str | None = None,
    service_level: float | None = None,
    grouping: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> InventoryIntelligenceResponse:
    if abc_class is not None and str(abc_class).upper() not in {"A", "B", "C"}:
        raise ValueError("abc_class must be one of A, B, or C.")
    if xyz_class is not None and str(xyz_class).upper() not in {"X", "Y", "Z"}:
        raise ValueError("xyz_class must be one of X, Y, or Z.")
    if risk_level is not None and str(risk_level).title() not in {"Low", "Medium", "High", "Critical"}:
        raise ValueError("risk_level must be one of Low, Medium, High, or Critical.")

    frame = _load_dataframe(product_id, store_id).copy()
    if category is not None:
        frame = frame[frame["category"].astype(str).str.lower() == str(category).lower()]
    if region is not None:
        frame = frame[frame["region"].astype(str).str.lower() == str(region).lower()]
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        return _empty_inventory_response()
    if service_level is None:
        service_level = 0.95
    if not 0 < float(service_level) < 1:
        raise ValueError("service_level must be between 0 and 1.")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["units_sold"] = pd.to_numeric(frame["units_sold"], errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    frame["inventory_on_hand"] = pd.to_numeric(frame["inventory_on_hand"], errors="coerce").fillna(0.0)
    frame["supplier_lead_time_days"] = pd.to_numeric(frame["supplier_lead_time_days"], errors="coerce").fillna(0.0)

    row_frame = build_inventory_intelligence_dataframe(frame, service_level=float(service_level)).copy()
    if row_frame.empty:
        return _empty_inventory_response()

    for field in ["current_inventory", "lead_time_days", "lead_time_demand", "reorder_point", "target_inventory", "recommended_order", "stockout_risk_score", "excess_inventory_units", "excess_inventory_percentage", "inventory_coverage", "health_score"]:
        row_frame[field] = pd.to_numeric(row_frame[field], errors="coerce").fillna(0.0)

    product_business = (
        frame.groupby("product_id", dropna=False, sort=True)
        .apply(lambda group: float((group["units_sold"] * group["price"]).sum()))
        .reset_index(name="business_value")
    )
    product_demand = (
        frame.groupby("product_id", dropna=False, sort=True)
        .apply(lambda group: pd.Series({"units_sold": float(group["units_sold"].sum())}))
        .reset_index()
    )
    product_abc = classify_abc(product_business, product_id_col="product_id", value_col="business_value")
    product_xyz = classify_xyz(frame[["product_id", "units_sold"]], product_id_col="product_id", demand_col="units_sold")
    abc_mapping = {item["product_id"]: item for item in product_abc}
    xyz_mapping = {item["product_id"]: item for item in product_xyz}

    row_frame["abc_class"] = row_frame["product_id"].map(lambda pid: abc_mapping.get(str(pid), {}).get("abc_class", "C"))
    row_frame["xyz_class"] = row_frame["product_id"].map(lambda pid: xyz_mapping.get(str(pid), {}).get("xyz_class", "X"))
    row_frame["business_value"] = row_frame["product_id"].map(lambda pid: abc_mapping.get(str(pid), {}).get("business_value", 0.0))

    row_frame = row_frame.copy()
    if product_id is not None:
        row_frame = row_frame[row_frame["product_id"] == str(product_id)]
    if store_id is not None:
        row_frame = row_frame[row_frame["store_id"] == int(store_id)]
    if abc_class is not None:
        row_frame = row_frame[row_frame["abc_class"].astype(str).str.upper() == str(abc_class).upper()]
    if xyz_class is not None:
        row_frame = row_frame[row_frame["xyz_class"].astype(str).str.upper() == str(xyz_class).upper()]
    if risk_level is not None:
        row_frame = row_frame[row_frame["stockout_risk_level"].astype(str).str.upper() == str(risk_level).upper()]
    if health_band is not None:
        row_frame = row_frame[row_frame["health_band"].astype(str).str.upper() == str(health_band).upper()]

    if row_frame.empty:
        return _empty_inventory_response()

    stockout_distribution = _distribution_from_values(row_frame["stockout_risk_level"].astype(str).tolist())
    excess_distribution = _distribution_from_values(row_frame["excess_risk_level"].astype(str).tolist())
    total_inventory_units = float(row_frame["current_inventory"].sum())
    stockout_risk_count = int((row_frame["stockout_risk_level"].isin(["High", "Critical"]) | row_frame["stockout_risk_level"].isin(["HIGH", "CRITICAL"])).sum())
    excess_inventory_count = int((row_frame["excess_risk_level"].isin(["Medium", "High"]) | row_frame["excess_risk_level"].isin(["MEDIUM", "HIGH"])).sum())
    critical_inventory_count = int((row_frame["health_band"].isin(["Risk", "Critical"]) | row_frame["health_band"].isin(["RISK", "CRITICAL"])).sum())
    average_health_score = float(row_frame["health_score"].mean()) if not row_frame.empty else 0.0
    abc_distribution = _distribution_from_values(row_frame["abc_class"].astype(str).tolist())
    xyz_distribution = _distribution_from_values(row_frame["xyz_class"].astype(str).tolist())

    abc_xyz_map = {}
    for _, item in row_frame[["product_id", "abc_class", "xyz_class", "business_value"]].drop_duplicates().iterrows():
        key = f"{str(item['abc_class']).upper()}{str(item['xyz_class']).upper()}"
        abc_xyz_map[key] = abc_xyz_map.get(key, 0) + 1
    abc_xyz_distribution = abc_xyz_map

    risk_matrix_data = product_store_risk_matrix(
        row_frame.to_dict(orient="records"),
        product_id=product_id,
        store_id=store_id,
        category=category,
        region=region,
        abc_class=abc_class,
        xyz_class=xyz_class,
        risk_level=risk_level,
    )

    opportunities = opportunity_detection(row_frame.to_dict(orient="records"))
    service_level_rows = []
    for service_entry in service_level_analysis(
        demand_std=float(row_frame["demand_std"].mean()) if not row_frame.empty else 0.0,
        lead_time_days=float(row_frame["lead_time_days"].mean()) if not row_frame.empty else 0.0,
        current_inventory=float(row_frame["current_inventory"].mean()) if not row_frame.empty else 0.0,
        service_levels=[0.90, 0.95, 0.98, 0.99],
    ):
        service_level_rows.append(
            InventoryServiceLevelComparison(
                service_level=service_entry["service_level"],
                z_score=service_entry["z_score"],
                safety_stock=service_entry["safety_stock"],
                reorder_point=service_entry["reorder_point"],
                target_inventory=service_entry["target_inventory"],
                recommended_order=service_entry["recommended_order"],
            )
        )

    inventory_health = InventoryHealthSummary(
        average_score=float(average_health_score),
        health_band_counts={band: int((row_frame["health_band"].astype(str).str.upper() == band.upper()).sum()) for band in ["Excellent", "Healthy", "Watch", "Risk", "Critical"]},
        top_critical_products=[str(item) for item in row_frame.sort_values("health_score", ascending=True).head(5)["product_id"].tolist()],
    )

    abc_xyz_summary = [
        InventoryABCXYZSummary(
            class_=segment,
            product_count=int(abc_xyz_map.get(segment, 0)),
            business_value=float(sum(item["business_value"] for _, item in row_frame[["product_id", "abc_class", "xyz_class", "business_value"]].drop_duplicates().iterrows() if f"{str(item['abc_class']).upper()}{str(item['xyz_class']).upper()}" == segment)),
            percentage_contribution=0.0,
            demand_variability=float(row_frame.loc[row_frame["abc_class"].astype(str).str.upper() + row_frame["xyz_class"].astype(str).str.upper() == segment, "coefficient_of_variation"].mean()) if any(str(row["abc_class"]).upper() + str(row["xyz_class"]).upper() == segment for _, row in row_frame.iterrows()) else 0.0,
        )
        for segment in ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
    ]

    for item in abc_xyz_summary:
        total_business = float(row_frame["business_value"].sum()) if not row_frame.empty else 0.0
        product_count = item.product_count
        item.percentage_contribution = float((item.business_value / total_business) * 100.0) if total_business else 0.0

    response = InventoryIntelligenceResponse(
        summary=InventoryIntelligenceSummary(
            total_products=int(row_frame["product_id"].nunique()),
            total_stores=int(row_frame["store_id"].nunique()),
            total_inventory_units=float(total_inventory_units),
            stockout_risk_count=int(stockout_risk_count),
            excess_inventory_count=int(excess_inventory_count),
            critical_inventory_count=int(critical_inventory_count),
            average_health_score=float(average_health_score),
            abc_distribution={key: int(value) for key, value in abc_distribution.items()},
            xyz_distribution={key: int(value) for key, value in xyz_distribution.items()},
            abc_xyz_distribution={key: int(value) for key, value in abc_xyz_distribution.items()},
        ),
        inventory_health=inventory_health,
        risk=InventoryRiskSummary(
            stockout_risk_distribution={key: int(value) for key, value in stockout_distribution.items()},
            excess_inventory_distribution={key: int(value) for key, value in excess_distribution.items()},
            risk_matrix_data=risk_matrix_data,
        ),
        abc_xyz=abc_xyz_summary,
        opportunities=[InventoryOpportunity(**item) for item in opportunities],
        service_level=service_level_rows,
    )
    return response
