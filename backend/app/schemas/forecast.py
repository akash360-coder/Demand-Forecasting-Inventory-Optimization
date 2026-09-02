from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    product_id: str = "P101"
    store_id: int = 1
    region: str | None = None
    forecast_horizon: int = Field(default=14, ge=1, le=180)
    date_from: str | None = None
    date_to: str | None = None


class ForecastPoint(BaseModel):
    date: str
    historical_demand: float | None = None
    forecast_demand: float
    lower_bound: float
    upper_bound: float


class ForecastSummary(BaseModel):
    product_id: str
    store_id: int
    region: str
    current_inventory: float
    forecast_total: float
    recommended_order: float
    stockout_risk: float
    excess_inventory_risk: float
    mae: float
    rmse: float
    mape: float
    wmape: float = 0.0


class ForecastResponse(BaseModel):
    summary: ForecastSummary
    points: list[ForecastPoint]


class DashboardSummary(BaseModel):
    demand_today: float
    inventory_risk: float
    average_forecast_error: float
    recommended_reorder: float
    risk_breakdown: dict[str, float]
    trend: list[dict[str, float | str]]


class ModelPerformanceMetrics(BaseModel):
    """Model evaluation metrics for a specific period."""

    mae: float
    rmse: float
    mape: float
    wmape: float


class ModelPerformanceResult(BaseModel):
    """Single model's performance results."""

    model_name: str
    validation_metrics: ModelPerformanceMetrics
    test_metrics: ModelPerformanceMetrics
    is_selected: bool = False


class ModelPerformanceReport(BaseModel):
    """Complete model performance comparison report."""

    timestamp: str
    dataset_rows: int
    feature_columns: list[str]
    validation_strategy: str
    selected_model: str
    selected_metrics: ModelPerformanceMetrics
    results: list[ModelPerformanceResult]

