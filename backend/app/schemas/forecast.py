from __future__ import annotations

from typing import Any

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


class InventoryResponse(BaseModel):
    product_id: str
    store_id: int
    current_inventory: float
    average_daily_demand: float
    lead_time_days: float
    lead_time_demand: float
    demand_std: float
    service_level: float
    safety_stock: float
    reorder_point: float
    target_inventory: float
    recommended_order_quantity: float
    inventory_coverage_days: float | None
    stockout_risk: float
    stockout_label: str
    excess_inventory_risk: float
    excess_label: str


class ExplanationFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class ExplainResponse(BaseModel):
    product_id: str
    store_id: int
    prediction: float
    top_features: list[ExplanationFeature]
    summary: str


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


class SimulationRequest(BaseModel):
    product_id: str = Field(min_length=1)
    store_id: int = Field(gt=0)
    forecast_horizon: int = Field(default=7, ge=1, le=30)
    price: float = Field(gt=0)
    promotion: bool = False
    holiday: bool = False
    lead_time_days: int = Field(gt=0, le=365)
    current_inventory: float = Field(ge=0)


class SimulationForecastPoint(BaseModel):
    date: str
    predicted_demand: float


class SimulationInventory(BaseModel):
    average_daily_demand: float
    lead_time_days: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    target_inventory: float
    current_inventory: float
    recommended_order: float
    coverage_days: float | None
    stockout_risk: float
    stockout_label: str
    excess_inventory_risk: float
    excess_inventory_label: str


class SimulationScenario(BaseModel):
    product_id: str
    store_id: int
    forecast_horizon: int
    price: float
    promotion: bool
    holiday: bool
    lead_time_days: int
    current_inventory: float


class SimulationImpact(BaseModel):
    demand_change: float
    demand_change_percent: float | None
    recommended_order_change: float
    reorder_point_change: float
    safety_stock_change: float


class SimulationExplanationFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str
    importance: float


class SimulationExplanation(BaseModel):
    prediction: float
    base_value: float
    features: list[SimulationExplanationFeature]
    summary: str
    sanity_check: bool


class SimulationRun(BaseModel):
    forecast_demand: float
    forecast: list[SimulationForecastPoint]
    inventory: SimulationInventory


class SimulationMetadata(BaseModel):
    model_name: str
    model_version: str
    generated_at: str


class SimulationResponse(BaseModel):
    scenario: SimulationScenario
    baseline: SimulationRun
    result: SimulationRun
    forecast: list[SimulationForecastPoint]
    inventory: SimulationInventory
    impact: SimulationImpact
    explanation: SimulationExplanation
    metadata: SimulationMetadata


class SimulationOptions(BaseModel):
    products: list[dict[str, str]]
    stores: list[dict[str, int | str]]


class ABCClassificationRecord(BaseModel):
    product_id: str
    business_value: float
    percentage_of_total: float
    cumulative_percentage: float
    abc_class: str


class XYZClassificationRecord(BaseModel):
    product_id: str
    mean_demand: float
    demand_std: float
    coefficient_of_variation: float
    xyz_class: str


class InventoryHealthSummary(BaseModel):
    average_score: float
    health_band_counts: dict[str, int]
    top_critical_products: list[str]


class InventoryRiskSummary(BaseModel):
    stockout_risk_distribution: dict[str, int]
    excess_inventory_distribution: dict[str, int]
    risk_matrix_data: dict[str, Any]


class InventoryABCXYZSummary(BaseModel):
    class_: str
    product_count: int
    business_value: float
    percentage_contribution: float
    demand_variability: float


class InventoryOpportunity(BaseModel):
    product_id: str
    store_id: int
    category: str | None = None
    region: str | None = None
    priority: str
    opportunity_type: str
    relevant_metric: str
    current_value: float
    threshold: float
    explanation: str


class InventoryServiceLevelComparison(BaseModel):
    service_level: float
    z_score: float
    safety_stock: float
    reorder_point: float
    target_inventory: float
    recommended_order: float


class InventoryIntelligenceSummary(BaseModel):
    total_products: int
    total_stores: int
    total_inventory_units: float
    stockout_risk_count: int
    excess_inventory_count: int
    critical_inventory_count: int
    average_health_score: float
    abc_distribution: dict[str, int]
    xyz_distribution: dict[str, int]
    abc_xyz_distribution: dict[str, int]


class InventoryIntelligenceResponse(BaseModel):
    summary: InventoryIntelligenceSummary
    inventory_health: InventoryHealthSummary
    risk: InventoryRiskSummary
    abc_xyz: list[InventoryABCXYZSummary]
    opportunities: list[InventoryOpportunity]
    service_level: list[InventoryServiceLevelComparison]


class MonitoringResponse(BaseModel):
    status: str
    generated_at: str
    data_quality: dict[str, Any]
    feature_drift: list[dict[str, Any]]
    target_drift: dict[str, Any]
    prediction_drift: dict[str, Any]
    model_performance: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    metadata: dict[str, Any]


class ModelMetadata(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    model_version: str
    artifact_path: str
    created_at: str
    feature_count: int
    feature_names: list[str]
    target_name: str
    metrics: dict[str, Any]
    status: str
    is_champion: bool
    is_production: bool
    parent_model_version: str | None = None
    training_run_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRegistryResponse(BaseModel):
    models: list[ModelMetadata]


class RetrainingResponse(BaseModel):
    training_run_id: str
    candidate_model_version: str
    champion_model_version: str
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float]
    wmape_improvement: float
    decision: str
    reason: str


class RollbackResponse(BaseModel):
    production_model_version: str
    model_name: str


class ForecastAccuracyResponse(BaseModel):
    summary: dict[str, Any]
    breakdowns: dict[str, list[dict[str, Any]]]
    trends: dict[str, list[dict[str, Any]]]
    bias: dict[str, Any]
    business_impact: dict[str, Any]
    best_worst: dict[str, Any]
    metadata: dict[str, Any]
