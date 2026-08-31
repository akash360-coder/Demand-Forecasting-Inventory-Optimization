from __future__ import annotations

import numpy as np
import pandas as pd


def detect_seasonal_period(df: pd.DataFrame) -> int:
    frame = df.sort_values("date").copy()
    if len(frame) < 14:
        return 7
    series = frame["units_sold"].astype(float).to_numpy()
    candidate_periods = [7, 14, 28]
    best_period = 7
    best_score = -np.inf

    for period in candidate_periods:
        if period >= len(series):
            continue
        left = series[:-period]
        right = series[period:]
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            continue
        corr = np.corrcoef(left, right)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_score:
            best_score = corr
            best_period = period

    return max(int(best_period), 1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "store_id", "date"]).copy()
    grouped = df.groupby(["product_id", "store_id"], sort=False)["units_sold"]

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped.transform(lambda s: s.shift(lag))

    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std().fillna(0.0)
        )

    df["price_change"] = df.groupby(["product_id", "store_id"], sort=False)["price"].transform(
        lambda s: s.pct_change().fillna(0.0)
    )
    df["price_index"] = df["price"] / df.groupby(["product_id", "store_id"], sort=False)["price"].transform("mean")
    df["inventory_on_hand_lag_1"] = df.groupby(["product_id", "store_id"], sort=False)["inventory_on_hand"].transform(
        lambda s: s.shift(1).ffill().fillna(0.0)
    )
    df["inventory_coverage"] = df["inventory_on_hand_lag_1"] / (df["lag_7"].fillna(1) + 1)
    if "supplier_lead_time_days" in df.columns:
        df["lead_time_days"] = df["supplier_lead_time_days"].fillna(0.0)
    else:
        df["lead_time_days"] = 0.0

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    required = [
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "day_of_week",
        "month",
        "week_of_year",
        "day_of_year",
        "quarter",
        "is_weekend",
        "promotion",
        "holiday",
        "price",
        "price_change",
        "inventory_on_hand_lag_1",
        "lead_time_days",
    ]
    valid = df.dropna(subset=required).reset_index(drop=True)
    return valid


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "day_of_week",
    "month",
    "week_of_year",
    "day_of_year",
    "quarter",
    "is_weekend",
    "promotion",
    "holiday",
    "price",
    "price_change",
    "inventory_on_hand_lag_1",
    "lead_time_days",
]
