from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.forecast import DashboardSummary, ForecastRequest, ForecastResponse, ModelPerformanceReport
from app.services.forecast_service import get_dashboard_summary, get_forecast_response, get_model_performance_report

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=ForecastResponse)
def forecast_endpoint(
    product_id: str = Query(default="P101", description="Product identifier"),
    store_id: int = Query(default=1, description="Store identifier"),
    region: str | None = Query(default=None, description="Region filter"),
    forecast_horizon: int = Query(default=14, ge=1, le=180),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> ForecastResponse:
    params = ForecastRequest(
        product_id=product_id,
        store_id=store_id,
        region=region,
        forecast_horizon=forecast_horizon,
        date_from=date_from,
        date_to=date_to,
    )
    try:
        return get_forecast_response(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/dashboard", response_model=DashboardSummary)
def dashboard_endpoint() -> DashboardSummary:
    try:
        return get_dashboard_summary()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models/performance", response_model=ModelPerformanceReport)
def model_performance_endpoint() -> ModelPerformanceReport:
    """Get model performance comparison across all trained models."""
    try:
        report = get_model_performance_report()
        return ModelPerformanceReport(**report)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

