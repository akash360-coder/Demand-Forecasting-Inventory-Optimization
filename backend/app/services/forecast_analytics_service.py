from __future__ import annotations

import pandas as pd

from app.services.forecast_service import _load_dataframe
from src.demand_intelligence.forecast_analytics import build_forecast_accuracy_analytics
from src.demand_intelligence.forecasting import load_production_model


def get_forecast_accuracy_response(
    product_id: str | None = None,
    store_id: int | None = None,
    category: str | None = None,
    region: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    group_by: str | None = None,
) -> dict:
    frame = _load_dataframe(product_id, store_id).copy()
    if category is not None:
        frame = frame[frame["category"] == category]
    if region is not None:
        frame = frame[frame["region"] == region]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("No data found for the requested analytics filters.")
    artifact = load_production_model()
    result = build_forecast_accuracy_analytics(frame, artifact=artifact)
    result["metadata"].update({
        "model_name": artifact.get("model_name"),
        "model_version": artifact.get("model_version"),
        "filters": {"product_id": product_id, "store_id": store_id, "category": category, "region": region, "start_date": start_date, "end_date": end_date},
    })
    if group_by:
        if group_by not in result["breakdowns"] and group_by not in result["trends"]:
            raise ValueError("group_by must be product, store, category, region, day, week, or month.")
        result["metadata"]["selected_group_by"] = group_by
    return result
