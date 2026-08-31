from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.demand_intelligence.feature_engineering import FEATURE_COLUMNS, build_feature_matrix


def _safe_percentage_error(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> np.ndarray:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    denom = np.abs(actual_arr)
    errors = np.zeros_like(actual_arr, dtype=float)
    mask = denom > 0
    errors[mask] = np.abs(actual_arr[mask] - predicted_arr[mask]) / denom[mask]
    return errors


def evaluate_predictions(actual: Iterable[float] | pd.Series, predicted: Iterable[float] | pd.Series) -> dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("Actual and predicted arrays must have the same length.")
    if actual_arr.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0}

    abs_errors = np.abs(actual_arr - predicted_arr)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(actual_arr - predicted_arr))))
    mape = float(np.mean(_safe_percentage_error(actual_arr, predicted_arr)) * 100.0)
    denom = np.sum(np.abs(actual_arr))
    wmape = float(np.sum(abs_errors) / denom * 100.0) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "wmape": wmape}


def compute_model_metrics(
    df: pd.DataFrame,
    model=None,
    feature_columns: list[str] | None = None,
    validation_frame: pd.DataFrame | None = None,
) -> dict[str, float]:
    frame = build_feature_matrix(df).copy()
    if len(frame) < 10:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "wmape": 0.0}

    columns = feature_columns or FEATURE_COLUMNS
    if validation_frame is not None:
        actual = validation_frame["units_sold"].astype(float).to_numpy()
        predicted = model.predict(validation_frame[columns]) if model is not None else RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ).fit(frame[columns], frame["units_sold"]).predict(validation_frame[columns])
        return evaluate_predictions(actual, predicted)

    if model is None:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        )
        split_index = max(1, int(len(frame) * 0.8))
        train = frame.iloc[:split_index].copy()
        test = frame.iloc[split_index:].copy()
        model.fit(train[columns], train["units_sold"])
        actual = test["units_sold"].astype(float).to_numpy()
        predicted = model.predict(test[columns])
    else:
        if "units_sold" not in df.columns:
            raise ValueError("The dataframe must include a units_sold column.")
        actual = df["units_sold"].astype(float).to_numpy()
        predicted = model.predict(df[columns])
    return evaluate_predictions(actual, predicted)
