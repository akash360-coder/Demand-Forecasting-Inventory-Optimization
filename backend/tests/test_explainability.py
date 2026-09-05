import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.demand_intelligence.explainability import ExplainabilityUnavailable, explain_model
from src.demand_intelligence.data_generation import ensure_dataset
from src.demand_intelligence.feature_engineering import build_feature_matrix
from src.demand_intelligence.forecasting import load_production_model


def _trained_model():
    features = pd.DataFrame({"lag_1": [1, 2, 3, 4], "price": [10, 9, 8, 7]})
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(features, [2, 4, 6, 8])
    return model, features.tail(1), ["lag_1", "price"]


def test_tree_model_returns_finite_ranked_shap_contributions():
    model, features, names = _trained_model()
    result = explain_model(model, features, names)
    assert np.isfinite(result["prediction"])
    assert result["sanity_check"]
    assert len(result["features"]) == 2
    assert all(np.isfinite(item.shap_value) for item in result["features"])
    assert abs(result["features"][0].shap_value) >= abs(result["features"][1].shap_value)


def test_invalid_features_are_rejected():
    model, features, names = _trained_model()
    with pytest.raises(ValueError, match="NaN"):
        explain_model(model, features.assign(lag_1=np.nan), names)


def test_baseline_model_is_not_given_fake_shap_values():
    with pytest.raises(ExplainabilityUnavailable, match="baseline"):
        explain_model("Naive", pd.DataFrame({"lag_1": [1.0]}), ["lag_1"])


def test_persisted_production_artifact_produces_real_shap_values():
    artifact = load_production_model()
    assert artifact["model_name"] in {"Random Forest", "XGBoost", "LightGBM"}
    frame = build_feature_matrix(ensure_dataset().query("product_id == 'P100' and store_id == 1"))
    result = explain_model(artifact["model"], frame[artifact["feature_columns"]].tail(1), artifact["feature_columns"])
    assert result["sanity_check"]
    assert np.isfinite(result["prediction"])
    assert result["features"]
    assert all(np.isfinite(item.shap_value) for item in result["features"])
