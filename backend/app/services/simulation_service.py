from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from app.core import database as database_module
from app.models import Product, Sales, Store
from app.schemas.forecast import (
    SimulationExplanation,
    SimulationForecastPoint,
    SimulationImpact,
    SimulationInventory,
    SimulationMetadata,
    SimulationOptions,
    SimulationRequest,
    SimulationResponse,
    SimulationRun,
    SimulationScenario,
)
from app.services.forecast_service import _load_dataframe
from src.demand_intelligence.explainability import explain_model
from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from src.demand_intelligence.forecasting import load_production_model
from src.demand_intelligence.inventory import inventory_recommendation


def _model_and_features() -> tuple[dict[str, Any], Any, list[str]]:
    artifact = load_production_model()
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    if not hasattr(model, "predict") or list(feature_columns or []) != FEATURE_COLUMNS:
        raise ValueError("The persisted production model is incompatible with the simulator feature contract.")
    return artifact, model, list(feature_columns)


def _future_forecast(
    history_frame: pd.DataFrame,
    horizon: int,
    model: Any,
    price: float,
    promotion: bool,
    holiday: bool,
    lead_time_days: int,
    current_inventory: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = build_feature_matrix(history_frame)
    if frame.empty:
        raise ValueError("Insufficient historical data to build a simulation forecast.")
    history = [max(0.0, float(value)) for value in frame["units_sold"].tolist()]
    last_date = pd.to_datetime(frame["date"].max())
    last_price = float(frame["price"].iloc[-1])
    if not isfinite(last_price) or last_price <= 0:
        raise ValueError("Historical price is invalid for simulation.")
    rows: list[dict[str, Any]] = []
    first_features: pd.DataFrame | None = None
    for offset in range(1, horizon + 1):
        date = last_date + pd.Timedelta(days=offset)
        past = history
        features = {
            "lag_1": past[-1] if past else 0.0,
            "lag_7": past[-7] if len(past) >= 7 else (past[-1] if past else 0.0),
            "lag_14": past[-14] if len(past) >= 14 else (past[-1] if past else 0.0),
            "rolling_mean_7": float(np.mean(past[-7:])) if past else 0.0,
            "rolling_mean_14": float(np.mean(past[-14:])) if past else 0.0,
            "rolling_std_7": float(np.std(past[-7:], ddof=0)) if len(past) >= 2 else 0.0,
            "day_of_week": int(date.dayofweek),
            "month": int(date.month),
            "week_of_year": int(date.isocalendar().week),
            "day_of_year": int(date.dayofyear),
            "quarter": int(date.quarter),
            "is_weekend": int(date.dayofweek >= 5),
            "promotion": int(promotion),
            "holiday": int(holiday),
            "price": float(price),
            "price_change": float(price / last_price - 1.0) if offset == 1 else 0.0,
            "inventory_on_hand_lag_1": float(current_inventory),
            "lead_time_days": float(lead_time_days),
        }
        feature_row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        if first_features is None:
            first_features = feature_row
        prediction = float(np.asarray(model.predict(feature_row)).reshape(-1)[0])
        if not isfinite(prediction):
            raise ValueError("The production model returned a non-finite simulation prediction.")
        prediction = max(0.0, prediction)
        rows.append({"date": date, "forecast_demand": prediction, "lower_bound": prediction, "upper_bound": prediction})
        history.append(prediction)
    assert first_features is not None
    return rows, first_features


def _inventory(frame: pd.DataFrame, forecast: list[dict[str, Any]], inventory: float, lead_time: int) -> SimulationInventory:
    context = frame.copy()
    context.loc[context.index[-1], "inventory_on_hand"] = inventory
    context.loc[context.index[-1], "supplier_lead_time_days"] = lead_time
    summary = inventory_recommendation(context, forecast)
    return SimulationInventory(
        average_daily_demand=float(summary["average_daily_demand"]),
        lead_time_days=float(lead_time),
        lead_time_demand=float(summary["lead_time_demand"]),
        safety_stock=float(summary["safety_stock"]),
        reorder_point=float(summary["reorder_point"]),
        target_inventory=float(summary["target_inventory"]),
        current_inventory=float(inventory),
        recommended_order=float(summary["recommended_order"]),
        coverage_days=float(summary["inventory_coverage_days"]) if summary["inventory_coverage_days"] is not None else None,
        stockout_risk=float(summary["stockout_risk"]),
        stockout_label=str(summary["stockout_label"]),
        excess_inventory_risk=float(summary["excess_inventory_risk"]),
        excess_inventory_label=str(summary["excess_label"]),
    )


def _run(frame: pd.DataFrame, request: SimulationRequest, model: Any, *, scenario: SimulationRequest) -> tuple[SimulationRun, pd.DataFrame]:
    forecast, explanation_features = _future_forecast(
        frame, request.forecast_horizon, model, scenario.price, scenario.promotion,
        scenario.holiday, scenario.lead_time_days, scenario.current_inventory,
    )
    inventory = _inventory(frame, forecast, scenario.current_inventory, scenario.lead_time_days)
    return (
        SimulationRun(
            forecast_demand=float(sum(point["forecast_demand"] for point in forecast)),
            forecast=[SimulationForecastPoint(date=point["date"].strftime("%Y-%m-%d"), predicted_demand=point["forecast_demand"]) for point in forecast],
            inventory=inventory,
        ),
        explanation_features,
    )


def get_simulation_options() -> SimulationOptions:
    database_module.configure_database()
    db = database_module.SessionLocal()
    try:
        products = [{"product_id": item.product_id, "name": item.name} for item in db.query(Product).filter(Product.active.is_(True)).order_by(Product.product_id).all()]
        stores = [{"store_id": int(item.store_id), "name": item.name, "region": item.region} for item in db.query(Store).filter(Store.active.is_(True)).order_by(Store.store_id).all()]
        return SimulationOptions(products=products, stores=stores)
    finally:
        db.close()


def run_simulation(request: SimulationRequest) -> SimulationResponse:
    frame = _load_dataframe(request.product_id, request.store_id)
    frame = frame[(frame["product_id"] == request.product_id) & (frame["store_id"] == request.store_id)].sort_values("date").copy()
    if frame.empty:
        raise ValueError(f"No data found for product {request.product_id} in store {request.store_id}.")
    artifact, model, _ = _model_and_features()
    latest = frame.iloc[-1]
    baseline = request.model_copy(update={
        "price": float(latest["price"]),
        "promotion": bool(latest["promotion"]),
        "holiday": bool(latest["holiday"]),
        "lead_time_days": int(latest["supplier_lead_time_days"]),
        "current_inventory": float(latest["inventory_on_hand"]),
    })
    baseline_run, _ = _run(frame, request, model, scenario=baseline)
    scenario_run, feature_row = _run(frame, request, model, scenario=request)
    explanation_data = explain_model(model, feature_row, FEATURE_COLUMNS)
    explanation = SimulationExplanation(
        prediction=explanation_data["prediction"],
        base_value=explanation_data["base_value"],
        features=[{**item.__dict__, "importance": abs(item.shap_value)} for item in explanation_data["features"]],
        summary=explanation_data["summary"],
        sanity_check=explanation_data["sanity_check"],
    )
    demand_change = scenario_run.forecast_demand - baseline_run.forecast_demand
    return SimulationResponse(
        scenario=SimulationScenario(**request.model_dump()),
        baseline=baseline_run,
        result=scenario_run,
        forecast=scenario_run.forecast,
        inventory=scenario_run.inventory,
        impact=SimulationImpact(
            demand_change=demand_change,
            demand_change_percent=(demand_change / baseline_run.forecast_demand * 100) if baseline_run.forecast_demand else None,
            recommended_order_change=scenario_run.inventory.recommended_order - baseline_run.inventory.recommended_order,
            reorder_point_change=scenario_run.inventory.reorder_point - baseline_run.inventory.reorder_point,
            safety_stock_change=scenario_run.inventory.safety_stock - baseline_run.inventory.safety_stock,
        ),
        explanation=explanation,
        metadata=SimulationMetadata(
            model_name=str(artifact.get("model_name", type(model).__name__)),
            model_version=str(artifact.get("model_version", "unknown")),
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
