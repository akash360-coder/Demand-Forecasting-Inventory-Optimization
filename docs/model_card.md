# Model Card

## Objective
Predict daily unit demand and estimate the inventory action needed to keep service levels high.

## Data
The project uses a synthetic retail sales dataset with realistic demand patterns across products, stores, and regions.

## Model
A Random Forest regressor is used for demand prediction, with lag features, calendar effects, and inventory coverage signals.

## Evaluation
The evaluation logic computes MAE, RMSE, and MAPE on a holdout validation split. Metrics are surfaced in the API and dashboard.

## Limitations
This is a synthetic portfolio-grade dataset and should not be used as the sole production forecasting baseline for real-world inventory planning without additional operational validation.
