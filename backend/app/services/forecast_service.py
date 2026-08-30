from __future__ import annotations

import pandas as pd

from app.core import database as database_module
from app.models import Sales
from app.schemas.forecast import DashboardSummary, ForecastRequest, ForecastResponse, ForecastSummary
from app.services.seed_demo import seed_demo_data
from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.evaluation import compute_model_metrics
from src.demand_intelligence.forecasting import generate_forecast_for_selection
from src.demand_intelligence.inventory import inventory_recommendation


def _load_dataframe() -> pd.DataFrame:
    seed_demo_data()
    database_module.configure_database()
    db = database_module.SessionLocal()
    try:
        sales = db.query(Sales).all()
        if sales:
            rows = [
                {
                    "date": item.date.isoformat(),
                    "product_id": item.product_id,
                    "product_name": item.product.name,
                    "store_id": item.store_id,
                    "region": item.store.region,
                    "category": item.product.category,
                    "units_sold": item.units_sold,
                    "price": item.price,
                    "promotion": item.promotion,
                    "holiday": item.holiday,
                    "inventory_on_hand": 0.0,
                    "supplier_lead_time_days": 0.0,
                }
                for item in sales
            ]
            if rows:
                return pd.DataFrame(rows)
    finally:
        db.close()
    return ensure_dataset()


def get_forecast_response(request: ForecastRequest) -> ForecastResponse:
    df = _load_dataframe()
    if request.region is not None:
        df = df[df["region"] == request.region]
    subset = df[(df["product_id"] == request.product_id) & (df["store_id"] == request.store_id)]
    if subset.empty:
        raise ValueError(f"No data found for product {request.product_id} in store {request.store_id}.")

    forecast = generate_forecast_for_selection(subset, horizon=request.forecast_horizon)
    summary = inventory_recommendation(subset, forecast)
    metrics = compute_model_metrics(subset)

    return ForecastResponse(
        summary=ForecastSummary(
            product_id=request.product_id,
            store_id=request.store_id,
            region=request.region or subset["region"].iloc[0],
            current_inventory=float(subset["inventory_on_hand"].iloc[-1]) if "inventory_on_hand" in subset.columns else 0.0,
            forecast_total=float(summary["forecast_total"]),
            recommended_order=float(summary["recommended_order"]),
            stockout_risk=float(summary["stockout_risk"]),
            excess_inventory_risk=float(summary["excess_inventory_risk"]),
            mae=float(metrics["mae"]),
            rmse=float(metrics["rmse"]),
            mape=float(metrics["mape"]),
        ),
        points=[
            {
                "date": point["date"].strftime("%Y-%m-%d"),
                "historical_demand": float(point["historical_demand"]) if point["historical_demand"] is not None else None,
                "forecast_demand": float(point["forecast_demand"]),
                "lower_bound": float(point["lower_bound"]),
                "upper_bound": float(point["upper_bound"]),
            }
            for point in forecast
        ],
    )


def get_dashboard_summary() -> DashboardSummary:
    df = _load_dataframe()
    forecast = generate_forecast_for_selection(df, horizon=14)
    summary = inventory_recommendation(df, forecast)
    trend = [
        {"date": item["date"].strftime("%Y-%m-%d"), "forecast": float(item["forecast_demand"])}
        for item in forecast[:7]
    ]
    return DashboardSummary(
        demand_today=float(summary["forecast_total"] / max(1, len(df["product_id"].unique()))),
        inventory_risk=float(summary["inventory_risk"]),
        average_forecast_error=float(summary["average_forecast_error"]),
        recommended_reorder=float(summary["recommended_order"]),
        risk_breakdown={
            "stockout": float(summary["stockout_risk"]),
            "excess": float(summary["excess_inventory_risk"]),
            "on_time": 1.0 - float(summary["stockout_risk"]),
        },
        trend=trend,
    )
