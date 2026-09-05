from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.demand_intelligence.evaluation import evaluate_predictions
from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from src.demand_intelligence.forecasting import load_production_model

MINIMUM_OBSERVATIONS = 7
BIAS_THRESHOLD = 0.05
GROUP_COLUMNS = {"product": ["product_id", "product_name"], "store": ["store_id"], "category": ["category"], "region": ["region"]}


def calculate_accuracy_metrics(actual: Iterable[float], forecast: Iterable[float]) -> dict[str, float]:
    actual_values = np.asarray(list(actual), dtype=float)
    forecast_values = np.asarray(list(forecast), dtype=float)
    if actual_values.shape != forecast_values.shape:
        raise ValueError("Actual and forecast values must have the same length.")
    if actual_values.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0, "bias": 0.0}
    if not np.isfinite(actual_values).all() or not np.isfinite(forecast_values).all():
        raise ValueError("Actual and forecast values must be finite.")
    metrics = evaluate_predictions(actual_values, forecast_values)
    total_actual = float(np.sum(np.abs(actual_values)))
    metrics["bias"] = float(np.sum(forecast_values - actual_values) / total_actual) if total_actual else 0.0
    return {key: float(value) if np.isfinite(value) else 0.0 for key, value in metrics.items()}


def build_forecast_error_dataset(
    df: pd.DataFrame,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"date", "product_id", "store_id", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Analytics data is missing columns: {', '.join(sorted(missing))}.")
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["units_sold"] = pd.to_numeric(frame["units_sold"], errors="coerce")
    valid = frame["date"].notna() & frame["product_id"].notna() & frame["store_id"].notna()
    valid &= np.isfinite(frame["units_sold"].to_numpy(dtype=float, na_value=np.nan))
    clean = frame.loc[valid].copy()
    if clean.empty:
        return pd.DataFrame(), {"input_rows": len(frame), "excluded_rows": len(frame), "valid_rows": 0}
    artifact = load_production_model() if model is None else {"model": model, "feature_columns": feature_columns or FEATURE_COLUMNS}
    columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    if list(columns) != FEATURE_COLUMNS:
        raise ValueError("The persisted production model has an incompatible feature contract.")
    feature_frame = build_feature_matrix(clean)
    if feature_frame.empty:
        return pd.DataFrame(), {"input_rows": len(frame), "excluded_rows": len(frame), "valid_rows": 0}
    predictions = np.asarray(artifact["model"].predict(feature_frame[FEATURE_COLUMNS]), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("The production model returned non-finite forecasts.")
    result = feature_frame[["date", "product_id", "store_id", "units_sold"]].copy()
    for column in ["product_name", "region", "category", "inventory_on_hand"]:
        if column in feature_frame:
            result[column] = feature_frame[column].values
    result = result.rename(columns={"units_sold": "actual_demand"})
    result["forecast_demand"] = predictions
    result["error"] = result["forecast_demand"] - result["actual_demand"]
    result["absolute_error"] = result["error"].abs()
    result["squared_error"] = result["error"].pow(2)
    result["absolute_percentage_error"] = np.where(result["actual_demand"] != 0, result["absolute_error"] / result["actual_demand"].abs() * 100, 0.0)
    result["signed_percentage_error"] = np.where(result["actual_demand"] != 0, result["error"] / result["actual_demand"].abs() * 100, 0.0)
    result["over_forecast"] = result["error"] > 0
    result["under_forecast"] = result["error"] < 0
    result["classification"] = np.select([result["over_forecast"], result["under_forecast"]], ["OVER_FORECAST", "UNDER_FORECAST"], default="EXACT")
    result = result.replace([np.inf, -np.inf], 0).fillna({"product_name": "Unknown", "region": "Unknown", "category": "Unknown"})
    return result.reset_index(drop=True), {"input_rows": len(frame), "excluded_rows": len(frame) - len(result), "valid_rows": len(result)}


def _aggregate(frame: pd.DataFrame, group_by: str, minimum_observations: int = MINIMUM_OBSERVATIONS) -> list[dict[str, Any]]:
    if group_by not in GROUP_COLUMNS:
        raise ValueError(f"Unsupported group_by value: {group_by}.")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_COLUMNS[group_by], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = calculate_accuracy_metrics(group["actual_demand"], group["forecast_demand"])
        item = dict(zip(GROUP_COLUMNS[group_by], keys))
        item.update(metrics)
        item.update({
            "actual_demand": float(group["actual_demand"].sum()),
            "forecast_demand": float(group["forecast_demand"].sum()),
            "observation_count": int(len(group)),
            "over_forecast_rate": float(group["over_forecast"].mean()),
            "under_forecast_rate": float(group["under_forecast"].mean()),
            "status": "INSUFFICIENT_DATA" if len(group) < minimum_observations else ("GOOD" if metrics["wmape"] < 20 else ("WARNING" if metrics["wmape"] < 35 else "POOR")),
        })
        rows.append(item)
    return rows


def _trend(frame: pd.DataFrame, period: str) -> list[dict[str, Any]]:
    dates = frame["date"].dt.to_period(period)
    grouped = frame.assign(period=dates).groupby("period", sort=True)
    rows = []
    for key, group in grouped:
        metrics = calculate_accuracy_metrics(group["actual_demand"], group["forecast_demand"])
        rows.append({"period": str(key), "date": str(key), "actual_demand": float(group["actual_demand"].sum()), "forecast_demand": float(group["forecast_demand"].sum()), **metrics, "observation_count": int(len(group))})
    return rows


def build_forecast_accuracy_analytics(
    df: pd.DataFrame,
    *,
    model: Any | None = None,
    artifact: dict[str, Any] | None = None,
    minimum_observations: int = MINIMUM_OBSERVATIONS,
    bias_threshold: float = BIAS_THRESHOLD,
) -> dict[str, Any]:
    if minimum_observations < 1 or bias_threshold < 0:
        raise ValueError("Analytics thresholds must be non-negative and observations must be positive.")
    if artifact is not None:
        model = artifact.get("model")
    errors, quality = build_forecast_error_dataset(df, model=model, feature_columns=(artifact or {}).get("feature_columns"))
    if errors.empty:
        raise ValueError("No valid observations are available for forecast accuracy analytics.")
    overall = calculate_accuracy_metrics(errors["actual_demand"], errors["forecast_demand"])
    over = errors["over_forecast"]
    under = errors["under_forecast"]
    bias_label = "OVER_FORECAST" if overall["bias"] > bias_threshold else ("UNDER_FORECAST" if overall["bias"] < -bias_threshold else "BALANCED")
    products = _aggregate(errors, "product", minimum_observations)
    stores = _aggregate(errors, "store", minimum_observations)
    categories = _aggregate(errors, "category", minimum_observations)
    regions = _aggregate(errors, "region", minimum_observations)
    def ranked(items: list[dict[str, Any]], reverse: bool) -> dict[str, Any] | None:
        eligible = [item for item in items if item["observation_count"] >= minimum_observations]
        return (sorted(eligible, key=lambda item: item["wmape"], reverse=reverse)[0] if eligible else None)
    inventory = errors.get("inventory_on_hand", pd.Series(0.0, index=errors.index)).astype(float)
    under_units = float(errors.loc[under, "actual_demand"].sub(errors.loc[under, "forecast_demand"]).sum())
    over_units = float(errors.loc[over, "forecast_demand"].sub(errors.loc[over, "actual_demand"]).sum())
    return {
        "summary": {**overall, "observation_count": len(errors), "over_forecast_rate": float(over.mean()), "under_forecast_rate": float(under.mean()), "exact_count": int((~over & ~under).sum())},
        "breakdowns": {"product": products, "store": stores, "category": categories, "region": regions},
        "trends": {"day": _trend(errors, "D"), "week": _trend(errors, "W"), "month": _trend(errors, "M")},
        "bias": {"label": bias_label, "threshold": bias_threshold, "over_forecast_count": int(over.sum()), "under_forecast_count": int(under.sum()), "exact_count": int((~over & ~under).sum()), "average_over_forecast_amount": float(errors.loc[over, "error"].mean()) if over.any() else 0.0, "average_under_forecast_amount": float(-errors.loc[under, "error"].mean()) if under.any() else 0.0},
        "business_impact": {"under_forecast_units": under_units, "over_forecast_units": over_units, "stockout_risk_count": int((inventory < errors["actual_demand"]).sum()), "excess_inventory_risk_count": int((inventory > errors["forecast_demand"] * 1.5).sum()), "interpretation": "Operational indicators, not validated monetary losses."},
        "best_worst": {"best_product": ranked(products, False), "worst_product": ranked(products, True), "best_store": ranked(stores, False), "worst_store": ranked(stores, True), "best_category": ranked(categories, False), "worst_category": ranked(categories, True), "best_region": ranked(regions, False), "worst_region": ranked(regions, True)},
        "metadata": {"generated_at": datetime.now(timezone.utc).isoformat(), **quality, "minimum_observations": minimum_observations, "bias_threshold": bias_threshold},
    }
