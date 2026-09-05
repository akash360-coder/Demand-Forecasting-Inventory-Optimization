from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np
import pandas as pd


class ExplainabilityUnavailable(ValueError):
    """Raised when the persisted model cannot produce a SHAP explanation."""


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    shap_value: float
    direction: str


def _display_name(feature: str) -> str:
    return feature.replace("_", " ").title()


def human_readable_summary(contributions: Sequence[FeatureContribution]) -> str:
    positive = [item.feature for item in contributions if item.shap_value > 0]
    negative = [item.feature for item in contributions if item.shap_value < 0]
    parts = []
    if positive:
        parts.append(f"{', '.join(positive[:2])} increased the forecast")
    if negative:
        parts.append(f"{', '.join(negative[:2])} reduced the forecast")
    return ". ".join(parts) + ("." if parts else "No material feature contribution was identified.")


def explain_model(model: Any, features: pd.DataFrame, feature_names: Sequence[str], top_n: int = 5) -> dict[str, Any]:
    if not hasattr(model, "predict"):
        raise ExplainabilityUnavailable("The persisted production model is a baseline and cannot be explained with SHAP.")
    if features.shape[0] != 1 or list(features.columns) != list(feature_names):
        raise ValueError("Explainability requires one row with the model's exact feature ordering.")
    values = features.to_numpy(dtype=float)
    if features.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("NaN and infinite feature values cannot be passed to SHAP.")
    prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
    if not isfinite(prediction):
        raise ExplainabilityUnavailable("The model returned a non-finite prediction.")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        base_value = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[0]
        shap_values = np.asarray(shap_values, dtype=float)
        if shap_values.ndim == 2:
            shap_values = shap_values[0]
        base_value = float(base_value)
    except Exception as exc:
        raise ExplainabilityUnavailable("SHAP could not explain the persisted model.") from exc
    if shap_values.ndim != 1 or len(shap_values) != len(feature_names) or not isfinite(base_value) or not np.isfinite(shap_values).all():
        raise ExplainabilityUnavailable("SHAP values do not align with the model features.")
    contributions = [
        FeatureContribution(_display_name(name), float(features.iloc[0][name]), float(value), "positive" if value >= 0 else "negative")
        for name, value in zip(feature_names, shap_values)
    ]
    contributions.sort(key=lambda item: abs(item.shap_value), reverse=True)
    return {
        "prediction": prediction,
        "base_value": base_value,
        "features": contributions[:max(1, top_n)],
        "summary": human_readable_summary(contributions[:max(1, top_n)]),
        "sanity_check": abs(base_value + float(shap_values.sum()) - prediction) < 1e-3,
    }
