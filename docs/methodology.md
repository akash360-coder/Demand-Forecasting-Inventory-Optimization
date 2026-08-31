# Forecasting Methodology

## Problem definition
The project forecasts daily retail demand by product and store. The target variable is `units_sold`, which is treated as a continuous daily demand outcome for operational planning and inventory decision support.

## Feature engineering
The feature set combines time, demand-history, and operational signals that are available before the forecast is generated:

- Time: `day_of_week`, `month`, `quarter`, `week_of_year`, `day_of_year`, `year`, `is_weekend`
- Demand lags: `lag_1`, `lag_7`, `lag_14`
- Rolling demand signals: `rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`
- Business variables: `promotion`, `holiday`, `price`, `price_change`, `lead_time_days`, `inventory_on_hand_lag_1`

All lag and rolling features are built with past observations only. For example, `rolling_mean_7(t)` uses only data observed before `t`, never the current or future target value.

## Leakage prevention
Random train/test splits are not used for the primary evaluation because they allow information from the future to leak into the training process. Instead, the model is evaluated with a strict time-series split:

1. Train period: earliest portion of the demand history
2. Validation period: immediately after training, in chronological order
3. Test period: final period reserved for unbiased model comparison

This ensures that no future values are used to predict earlier points.

## Baselines and machine learning models
The forecasting pipeline includes:

- Naive baseline
- Seasonal naive baseline
- Moving-average baseline
- Random Forest regressor
- XGBoost regressor
- LightGBM regressor

The baselines are intentionally simple and use only historical demand. The ML models use the engineered feature matrix for multi-step forecasting at a daily horizon.

## Validation and model selection
The project reports MAE, RMSE, MAPE, and WMAPE. These metrics are computed on a time-based validation set and compared across models. The production model is selected using the validation WMAPE, with lower values indicating better demand forecasting performance.

## Model persistence and inference
The trained production model is stored using joblib and loaded for inference instead of retraining on every API request. This keeps training and inference separate, which is important for production reliability and reproducibility.

## Limitations
This portfolio implementation uses a synthetic, reproducible retail dataset rather than a private enterprise ledger. While this keeps the project transparent and portable, it means the observed demand patterns are generated rather than directly sourced from a company’s historical operations.
