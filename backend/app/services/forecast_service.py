from __future__ import annotations

import pandas as pd

from app.core import database as database_module
from app.models import Inventory, Sales
from app.schemas.forecast import DashboardSummary, ForecastRequest, ForecastResponse, ForecastSummary, InventoryResponse
from app.services.seed_demo import seed_demo_data
from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.evaluation import compute_model_metrics
from src.demand_intelligence.forecasting import generate_forecast_for_selection
from src.demand_intelligence.inventory import inventory_recommendation


def _load_dataframe(product_id: str | None = None, store_id: int | None = None) -> pd.DataFrame:
    seed_demo_data()
    database_module.configure_database()
    db = database_module.SessionLocal()
    try:
        sales_query = db.query(Sales)
        inventory_query = db.query(Inventory)
        if product_id is not None:
            sales_query = sales_query.filter(Sales.product_id == product_id)
            inventory_query = inventory_query.filter(Inventory.product_id == product_id)
        if store_id is not None:
            sales_query = sales_query.filter(Sales.store_id == store_id)
            inventory_query = inventory_query.filter(Inventory.store_id == store_id)
        sales = sales_query.all()
        inventory = {(item.product_id, item.store_id): item for item in inventory_query.all()}
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
                    "inventory_on_hand": float(inventory.get((item.product_id, item.store_id)).inventory_on_hand) if (item.product_id, item.store_id) in inventory else 0.0,
                    "supplier_lead_time_days": float(inventory.get((item.product_id, item.store_id)).lead_time_days) if (item.product_id, item.store_id) in inventory else 0.0,
                }
                for item in sales
            ]
            if rows:
                return pd.DataFrame(rows)
    finally:
        db.close()
    return ensure_dataset()


def get_forecast_response(request: ForecastRequest) -> ForecastResponse:
    df = _load_dataframe(request.product_id, request.store_id)
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
            wmape=float(metrics["wmape"]),
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


def get_inventory_response(request: ForecastRequest) -> InventoryResponse:
    df = _load_dataframe(request.product_id, request.store_id)
    if request.region is not None:
        df = df[df["region"] == request.region]
    subset = df[(df["product_id"] == request.product_id) & (df["store_id"] == request.store_id)]
    if subset.empty:
        raise ValueError(f"No data found for product {request.product_id} in store {request.store_id}.")
    forecast = generate_forecast_for_selection(subset, horizon=request.forecast_horizon)
    summary = inventory_recommendation(subset, forecast)
    return InventoryResponse(
        product_id=request.product_id,
        store_id=request.store_id,
        current_inventory=float(subset["inventory_on_hand"].iloc[-1]),
        average_daily_demand=float(summary["average_daily_demand"]),
        lead_time_days=float(summary["lead_time_days"]),
        lead_time_demand=float(summary["lead_time_demand"]),
        demand_std=float(summary["demand_std"]),
        service_level=float(summary["service_level"]),
        safety_stock=float(summary["safety_stock"]),
        reorder_point=float(summary["reorder_point"]),
        target_inventory=float(summary["target_inventory"]),
        recommended_order_quantity=float(summary["recommended_order"]),
        inventory_coverage_days=float(summary["inventory_coverage_days"]) if summary["inventory_coverage_days"] is not None else None,
        stockout_risk=float(summary["stockout_risk"]),
        stockout_label=str(summary["stockout_label"]),
        excess_inventory_risk=float(summary["excess_inventory_risk"]),
        excess_label=str(summary["excess_label"]),
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


def get_model_performance_report() -> dict:
    """Get comprehensive model performance comparison."""
    from datetime import datetime
    from src.demand_intelligence.forecasting import run_experiment
    from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS

    df = _load_dataframe()
    experiment = run_experiment(df)

    results = []
    for result in experiment["results"]:
        results.append(
            {
                "model_name": result["model_name"],
                "validation_metrics": {
                    "mae": float(result["validation_metrics"]["mae"]),
                    "rmse": float(result["validation_metrics"]["rmse"]),
                    "mape": float(result["validation_metrics"]["mape"]),
                    "wmape": float(result["validation_metrics"]["wmape"]),
                },
                "test_metrics": {
                    "mae": float(result["test_metrics"]["mae"]),
                    "rmse": float(result["test_metrics"]["rmse"]),
                    "mape": float(result["test_metrics"]["mape"]),
                    "wmape": float(result["test_metrics"]["wmape"]),
                },
                "is_selected": result["model_name"] == experiment["selected_model"],
            }
        )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_rows": experiment["dataset_rows"],
        "feature_columns": FEATURE_COLUMNS,
        "validation_strategy": experiment["validation_strategy"],
        "selected_model": experiment["selected_model"],
        "selected_metrics": {
            "mae": float(experiment["selected_validation_metrics"]["mae"]),
            "rmse": float(experiment["selected_validation_metrics"]["rmse"]),
            "mape": float(experiment["selected_validation_metrics"]["mape"]),
            "wmape": float(experiment["selected_validation_metrics"]["wmape"]),
        },
        "results": results,
    }
