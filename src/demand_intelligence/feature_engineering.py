from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "store_id", "date"]).copy()
    grouped = df.groupby(["product_id", "store_id"], sort=False)["units_sold"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped.transform(lambda s: s.shift(lag))
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["price_index"] = df["price"] / df["price"].mean()
    df["inventory_coverage"] = df["inventory_on_hand"] / (df["lag_7"].fillna(1) + 1)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    return df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)
