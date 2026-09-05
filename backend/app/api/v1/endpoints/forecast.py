from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.demand_intelligence.explainability import ExplainabilityUnavailable
from app.schemas.forecast import DashboardSummary, ExplainResponse, ForecastRequest, ForecastResponse, InventoryResponse, ModelPerformanceReport, SimulationOptions, SimulationRequest, SimulationResponse
from app.services.forecast_service import get_dashboard_summary, get_explanation_response, get_forecast_response, get_inventory_response, get_model_performance_report
from app.services.simulation_service import get_simulation_options, run_simulation
from app.services.monitoring_service import get_monitoring_response
from app.schemas.forecast import ForecastAccuracyResponse, InventoryIntelligenceResponse, ModelMetadata, ModelRegistryResponse, MonitoringResponse, RetrainingResponse, RollbackResponse
from src.demand_intelligence.model_registry import get_model, list_models, rollback_to_model
from app.services.retraining_service import retrain_model
from app.services.forecast_analytics_service import get_forecast_accuracy_response
from app.services.inventory_intelligence_service import get_inventory_intelligence_response

router = APIRouter(tags=["forecast"])


@router.get("/analytics/forecast-accuracy", response_model=ForecastAccuracyResponse, tags=["analytics"])
def forecast_accuracy_endpoint(
    product_id: str | None = Query(default=None),
    store_id: int | None = Query(default=None, gt=0),
    category: str | None = Query(default=None),
    region: str | None = Query(default=None),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    group_by: str | None = Query(default=None),
) -> ForecastAccuracyResponse:
    try:
        return ForecastAccuracyResponse(**get_forecast_accuracy_response(product_id, store_id, category, region, start_date, end_date, group_by))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/analytics/inventory-intelligence", response_model=InventoryIntelligenceResponse, tags=["analytics"])
def inventory_intelligence_endpoint(
    product_id: str | None = Query(default=None),
    store_id: int | None = Query(default=None, gt=0),
    category: str | None = Query(default=None),
    region: str | None = Query(default=None),
    abc_class: str | None = Query(default=None),
    xyz_class: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    health_band: str | None = Query(default=None),
    service_level: float | None = Query(default=0.95, ge=0.0, le=1.0),
    grouping: str | None = Query(default=None),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> InventoryIntelligenceResponse:
    try:
        return get_inventory_intelligence_response(
            product_id=product_id,
            store_id=store_id,
            category=category,
            region=region,
            abc_class=abc_class,
            xyz_class=xyz_class,
            risk_level=risk_level,
            health_band=health_band,
            service_level=service_level,
            grouping=grouping,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@router.get("/inventory", response_model=InventoryResponse)
def inventory_endpoint(
    product_id: str = Query(default="P101"),
    store_id: int = Query(default=1),
    region: str | None = Query(default=None),
    forecast_horizon: int = Query(default=14, ge=1, le=180),
) -> InventoryResponse:
    try:
        return get_inventory_response(ForecastRequest(product_id=product_id, store_id=store_id, region=region, forecast_horizon=forecast_horizon))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/explain", response_model=ExplainResponse)
def explain_endpoint(
    product_id: str = Query(default="P101"),
    store_id: int = Query(default=1),
    region: str | None = Query(default=None),
    forecast_horizon: int = Query(default=14, ge=1, le=180),
) -> ExplainResponse:
    try:
        return ExplainResponse(**get_explanation_response(ForecastRequest(product_id=product_id, store_id=store_id, region=region, forecast_horizon=forecast_horizon)))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Forecast explanation is temporarily unavailable.") from exc


@router.get("/models/performance", response_model=ModelPerformanceReport)
def model_performance_endpoint() -> ModelPerformanceReport:
    """Get model performance comparison across all trained models."""
    try:
        report = get_model_performance_report()
        return ModelPerformanceReport(**report)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/simulation/options", response_model=SimulationOptions)
def simulation_options_endpoint() -> SimulationOptions:
    try:
        return get_simulation_options()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate", response_model=SimulationResponse)
def simulate_endpoint(request: SimulationRequest) -> SimulationResponse:
    try:
        return run_simulation(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="The production forecasting model is unavailable.") from exc
    except ExplainabilityUnavailable as exc:
        raise HTTPException(status_code=503, detail="Scenario explanation is temporarily unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring_endpoint(
    product_id: str | None = Query(default=None),
    store_id: int | None = Query(default=None, gt=0),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
) -> MonitoringResponse:
    try:
        return get_monitoring_response(product_id, store_id, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models", response_model=ModelRegistryResponse)
def models_endpoint() -> ModelRegistryResponse:
    return ModelRegistryResponse(models=[ModelMetadata(**item) for item in list_models()])


@router.get("/models/production", response_model=ModelMetadata)
def production_model_endpoint() -> ModelMetadata:
    return ModelMetadata(**next(item for item in list_models() if item["is_production"]))


@router.post("/models/retrain", response_model=RetrainingResponse)
def retrain_endpoint() -> RetrainingResponse:
    try:
        return RetrainingResponse(**retrain_model())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{model_version}/rollback", response_model=RollbackResponse)
def rollback_endpoint(model_version: str) -> RollbackResponse:
    try:
        record = rollback_to_model(model_version)
        return RollbackResponse(production_model_version=record["model_version"], model_name=record["model_name"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
