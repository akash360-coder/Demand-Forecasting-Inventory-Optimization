from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.schemas.forecast import MonitoringResponse
from app.services.forecast_service import _load_dataframe
from src.demand_intelligence.evaluation import evaluate_predictions
from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from src.demand_intelligence.forecasting import load_production_model
from src.demand_intelligence.monitoring import PSI_CRITICAL, PSI_WARNING, categorical_drift, data_quality, distribution_summary, drift_status, performance_comparison, psi


def get_monitoring_response(product_id: str | None = None, store_id: int | None = None, start_date: str | None = None, end_date: str | None = None) -> MonitoringResponse:
    frame = _load_dataframe(product_id, store_id).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("The monitoring window contains no data.")
    reference_all = _load_dataframe(product_id, store_id).copy()
    reference_all["date"] = pd.to_datetime(reference_all["date"])
    reference = reference_all.sort_values("date").iloc[: max(1, int(len(reference_all) * .8))]
    current = frame.sort_values("date")
    model_artifact = load_production_model()
    model = model_artifact.get("model")
    columns = model_artifact.get("feature_columns")
    if list(columns or []) != FEATURE_COLUMNS or not hasattr(model, "predict"):
        raise ValueError("The persisted production model is incompatible with monitoring.")
    ref_features = build_feature_matrix(reference)
    cur_features = build_feature_matrix(current)
    if ref_features.empty or cur_features.empty:
        raise ValueError("Insufficient data to build monitoring features.")
    reference_predictions = model.predict(ref_features[FEATURE_COLUMNS])
    current_predictions = model.predict(cur_features[FEATURE_COLUMNS])
    feature_drift = []
    for feature in FEATURE_COLUMNS:
        score = categorical_drift(ref_features[feature], cur_features[feature]) if feature in {"promotion", "holiday"} else psi(ref_features[feature], cur_features[feature])
        feature_drift.append({"feature": feature, "metric": "PSI", "score": score, "status": drift_status(score), "reference_statistics": distribution_summary(ref_features[feature]), "current_statistics": distribution_summary(cur_features[feature])})
    target_score = psi(ref_features["units_sold"], cur_features["units_sold"])
    prediction_score = psi(reference_predictions, current_predictions)
    performance = performance_comparison(cur_features["units_sold"], current_predictions, model_artifact.get("metrics", {}))
    quality = data_quality(current)
    alerts: list[dict[str, Any]] = []
    if quality["status"] != "HEALTHY":
        alerts.append({"severity": quality["status"], "category": "DATA_QUALITY", "message": "Current monitoring data contains quality issues.", "feature": None, "metric": "data_quality", "score": float(quality["invalid_count"] + quality["non_finite_count"])})
    for item in feature_drift:
        if item["status"] != "HEALTHY":
            alerts.append({"severity": item["status"], "category": "FEATURE_DRIFT", "message": f"{item['feature']} distribution drift detected.", "feature": item["feature"], "metric": item["metric"], "score": item["score"]})
    for category, score in (("TARGET_DRIFT", target_score), ("PREDICTION_DRIFT", prediction_score)):
        if drift_status(score) != "HEALTHY":
            alerts.append({"severity": drift_status(score), "category": category, "message": f"{category.replace('_', ' ').title()} detected.", "feature": "units_sold" if category == "TARGET_DRIFT" else None, "metric": "PSI", "score": score})
    for item in performance:
        if item["status"] != "HEALTHY":
            alerts.append({"severity": item["status"], "category": "MODEL_PERFORMANCE", "message": f"{item['metric']} degradation detected.", "feature": None, "metric": item["metric"], "score": item["change"]})
    overall = "CRITICAL" if any(item["severity"] == "CRITICAL" for item in alerts) else ("WARNING" if alerts else "HEALTHY")
    return MonitoringResponse(status=overall, generated_at=datetime.now(timezone.utc).isoformat(), data_quality=quality, feature_drift=feature_drift, target_drift={"score": target_score, "status": drift_status(target_score), "reference_statistics": distribution_summary(ref_features["units_sold"]), "current_statistics": distribution_summary(cur_features["units_sold"])}, prediction_drift={"score": prediction_score, "status": drift_status(prediction_score), "reference_statistics": distribution_summary(reference_predictions), "current_statistics": distribution_summary(current_predictions)}, model_performance=performance, alerts=alerts, metadata={"model_name": model_artifact.get("model_name"), "model_version": model_artifact.get("model_version"), "reference_period": [str(reference["date"].min().date()), str(reference["date"].max().date())], "monitoring_period": [str(current["date"].min().date()), str(current["date"].max().date())], "reference_rows": len(reference), "monitoring_rows": len(current)})
