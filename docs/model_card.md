# Model Card

## Objective
Predict daily unit demand and estimate the inventory action needed to keep service levels high.

## Data
The project uses a synthetic retail sales dataset with realistic demand patterns across products, stores, and regions.

## Model
The persisted artifact is a LightGBM regressor selected by the forecasting experiment using validation WMAPE. Tree-based artifacts use the same ordered feature vector of 18 lag, calendar, business, and inventory features. Baseline models remain available for benchmarking.

## Explainability
The `/api/v1/explain` endpoint uses SHAP TreeExplainer for persisted tree-based models and ranks the five largest absolute local contributions. Positive values increase the prediction relative to the expected model output; negative values decrease it. SHAP describes model behavior and does not establish causation. The current Naive artifact cannot produce valid feature contributions, so the endpoint returns an explicit unavailable response rather than fabricated values.

## Evaluation
The evaluation logic computes MAE, RMSE, and MAPE on a holdout validation split. Metrics are surfaced in the API and dashboard.

## Accuracy reporting
Forecast accuracy intelligence uses the persisted production artifact and historical actual demand to report MAE, RMSE, zero-safe MAPE, WMAPE, signed bias, and segmented trends. Invalid records are counted in metadata. Under/over-forecast and stockout/excess fields are operational indicators, not validated financial outcomes.

## Inventory intelligence
The inventory intelligence analytics layer is a decision-support view built from the same demand and inventory data. It segments products with ABC and XYZ logic, combines them into an ABC-XYZ matrix, scores portfolio health on a 0-100 scale, estimates stockout and excess-risk posture, ranks opportunities, and shows service-level trade-offs at 90%, 95%, 98%, and 99%. These outputs are operational summaries for planning discussions and do not guarantee future sales, stockouts, or financial outcomes.

## Limitations
This is a synthetic portfolio-grade dataset and should not be used as the sole production forecasting baseline for real-world inventory planning without additional operational validation.
### Simulation inference

What-if requests use the persisted production artifact and the exact saved
feature ordering. Scenario variables are applied only to future feature rows;
historical demand and price context is not rewritten. The endpoint fails
explicitly when the artifact, feature contract, prediction, or SHAP output is
invalid.

## Monitoring limitations

Monitoring provides signals for data quality, distribution changes, and
performance degradation. Distribution drift does not automatically prove model
failure or causal concept drift. Historical monitoring runs are not stored.

## Model lifecycle

The production LightGBM artifact is registered as the champion. Explicit
retraining creates a challenger, evaluates it with chronological data and
existing metrics, and promotes it only when the configured WMAPE improvement
threshold and compatibility checks pass. Registry-managed artifacts support
safe rollback; monitoring does not automatically retrain or replace models.
