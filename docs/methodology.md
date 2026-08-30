# Methodology

The forecasting workflow is designed for operational decision-making:

1. Create a realistic sales dataset with seasonality, promotions, holidays, store effects, and product trends.
2. Engineer lag and rolling features to capture autocorrelation and demand momentum.
3. Train a tree-based regressor to estimate demand for the next planning horizon.
4. Quantify forecast uncertainty with bounded prediction bands.
5. Convert forecasted demand into inventory actions using service-level and lead-time assumptions.

This approach is production-minded but intentionally compact enough to be reproducible within a portfolio project.
