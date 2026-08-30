from __future__ import annotations

import numpy as np

from src.demand_intelligence.feature_engineering import build_feature_matrix
from src.demand_intelligence.forecasting import FEATURE_COLUMNS, _fit_model


def compute_model_metrics(df):
    frame = build_feature_matrix(df).copy()
    if len(frame) < 10:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    split_index = max(1, int(len(frame) * 0.8))
    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()
    model = _fit_model(train)
    preds = model.predict(test[FEATURE_COLUMNS])
    mae = float(np.mean(np.abs(test["units_sold"] - preds)))
    rmse = float(np.sqrt(np.mean((test["units_sold"] - preds) ** 2)))
    mape = float(np.mean(np.abs((test["units_sold"] - preds) / (np.where(test["units_sold"] == 0, 1, test["units_sold"]))))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}
