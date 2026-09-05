from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.demand_intelligence.evaluation import evaluate_predictions

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


def _finite(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    return values[np.isfinite(values)]


def psi(reference: Iterable[float], current: Iterable[float], bins: int = 10) -> float:
    ref = _finite(reference)
    cur = _finite(current)
    if not len(ref) or not len(cur):
        return 0.0
    if np.all(ref == ref[0]):
        edges = np.array([ref[0] - 0.5, ref[0] + 0.5])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([ref.min() - 0.5, ref.max() + 0.5])
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_rate = np.maximum(ref_counts / len(ref), 1e-6)
    cur_rate = np.maximum(cur_counts / len(cur), 1e-6)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def drift_status(score: float) -> str:
    if score >= PSI_CRITICAL:
        return "CRITICAL"
    if score >= PSI_WARNING:
        return "WARNING"
    return "HEALTHY"


def categorical_drift(reference: Iterable[Any], current: Iterable[Any]) -> float:
    ref = pd.Series(list(reference)).fillna("<MISSING>").astype(str)
    cur = pd.Series(list(current)).fillna("<MISSING>").astype(str)
    categories = sorted(set(ref) | set(cur))
    ref_rates = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_rates = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    ref_rates = np.maximum(ref_rates, 1e-6)
    cur_rates = np.maximum(cur_rates, 1e-6)
    return float(np.sum((cur_rates - ref_rates) * np.log(cur_rates / ref_rates)))


def data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    monitored = [column for column in frame.columns if column in {"units_sold", "price", "promotion", "holiday", "inventory_on_hand", "supplier_lead_time_days"}]
    missing = {column: {"missing_count": int(frame[column].isna().sum()), "missing_rate": float(frame[column].isna().mean())} for column in monitored}
    invalid = {
        "negative_price": int((frame["price"] < 0).sum()) if "price" in frame else 0,
        "negative_inventory": int((frame["inventory_on_hand"] < 0).sum()) if "inventory_on_hand" in frame else 0,
        "invalid_lead_time": int((frame["supplier_lead_time_days"] <= 0).sum()) if "supplier_lead_time_days" in frame else 0,
        "invalid_promotion": int((~frame["promotion"].isin([0, 1])).sum()) if "promotion" in frame else 0,
        "invalid_holiday": int((~frame["holiday"].isin([0, 1])).sum()) if "holiday" in frame else 0,
        "negative_demand": int((frame["units_sold"] < 0).sum()) if "units_sold" in frame else 0,
    }
    numeric = frame.select_dtypes(include=[np.number])
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    duplicate_count = int(frame.duplicated().sum())
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()) if len(frame) and "date" in frame else None,
        "date_max": str(pd.to_datetime(frame["date"]).max().date()) if len(frame) and "date" in frame else None,
        "missing": missing,
        "duplicate_count": duplicate_count,
        "duplicate_rate": float(duplicate_count / len(frame)) if len(frame) else 0.0,
        "invalid_values": invalid,
        "invalid_count": int(sum(invalid.values())),
        "non_finite_count": non_finite,
        "status": "CRITICAL" if non_finite or sum(invalid.values()) else ("WARNING" if duplicate_count or any(item["missing_count"] for item in missing.values()) else "HEALTHY"),
    }


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    values = _finite(values)
    if not len(values):
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q75": 0.0}
    return {name: float(value) for name, value in {"mean": np.mean(values), "median": np.median(values), "std": np.std(values), "min": np.min(values), "max": np.max(values), "q25": np.quantile(values, .25), "q75": np.quantile(values, .75)}.items()}


def performance_comparison(actual: Iterable[float], predicted: Iterable[float], reference: dict[str, float]) -> list[dict[str, Any]]:
    current = evaluate_predictions(actual, predicted)
    entries = []
    for key in ("mae", "rmse", "mape", "wmape"):
        ref = float(reference.get(key, 0.0))
        value = float(current[key])
        change = value - ref
        entries.append({"metric": key.upper(), "reference_value": ref, "current_value": value, "change": change, "change_percent": (change / ref * 100) if ref else None, "status": "CRITICAL" if ref and change / ref >= .25 else ("WARNING" if ref and change / ref >= .10 else "HEALTHY")})
    return entries
