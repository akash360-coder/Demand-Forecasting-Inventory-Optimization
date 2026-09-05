# Demand Intelligence: AI-Powered Demand Forecasting & Inventory Optimization Platform

This project is a production-style portfolio application for retail demand forecasting and inventory optimization. It combines data generation, feature engineering, forecasting, a FastAPI backend, and a Next.js dashboard.

## Forecast Accuracy Intelligence

`GET /api/v1/analytics/forecast-accuracy` evaluates the persisted production model against historical actual demand. It reports MAE, RMSE, zero-safe MAPE, WMAPE, and signed bias (`forecast - actual` divided by total actual demand), segmented by product, store, category, region, and day/week/month. Under- and over-forecast units plus stockout/excess-inventory observations are operational indicators, not causal or monetary impact estimates. Groups with fewer than seven observations are marked insufficient.

## Inventory Intelligence

`GET /api/v1/analytics/inventory-intelligence` provides decision-support inventory analytics derived from the current portfolio. It classifies products into ABC and XYZ segments, combines them into an ABC-XYZ matrix, scores inventory health on a 0-100 scale, identifies stockout and excess-risk posture, ranks opportunities, and reports service-level trade-offs at 90%, 95%, 98%, and 99%. All outputs are descriptive operational indicators that do not guarantee future business results.

- ABC methodology: rank products by cumulative business value and assign A/B/C bands based on total contribution thresholds.
- XYZ methodology: measure coefficient of variation by product and classify X/Y/Z based on variability.
- ABC-XYZ matrix: combine value and variability to surface high-value, stable, and volatile planning scenarios.
- Inventory health score: blends stockout exposure, excess exposure, coverage, and variability to create a single portfolio health rating.
- Stockout and excess inventory: directional operational risk signals used to prioritize replenishment and reduction actions.
- Opportunity ranking: surfaces the highest-priority actions across the filtered portfolio, such as replenishment or liquidation opportunities.
- Service-level analytics: show how target service levels change safety stock, reorder points, and recommended order quantities.
- Limitations: this is decision support for inventory improvement, not a guarantee of future sales, margin, or supply outcomes.

## Business problem
The app answers:
- demand by product/store/region
- rising and falling product demand
- stockout risk
- excess inventory
- reorder recommendations
- expected inventory risk

## Project structure

```text
.
├── backend/
│   ├── app/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
├── frontend/
├── notebooks/
├── src/
├── docker-compose.yml
├── .env.example
├── README.md
└── ...
```

## Database configuration
The application uses PostgreSQL as the recommended runtime database. SQLite is supported only for explicit local development by setting `DATABASE_MODE=sqlite` in the environment. The app does not silently fall back to SQLite when PostgreSQL settings are expected or partially configured.

## Data note
This project uses a realistic, synthetic retail dataset to remain reproducible without depending on a private or unavailable business dataset. The fields `date`, `product_id`, `store_id`, `region`, `category`, and `units_sold` reflect realistic public retail demand patterns. Operational fields such as `price`, `promotion`, `holiday`, `inventory_on_hand`, and `supplier_lead_time_days` are generated to simulate a business environment for inventory optimization. This is explicitly documented in `docs/data_dictionary.md` so the distinction between real, derived, and simulated fields is clear.

## Quick start

### Python backend
```bash
cd backend
pip install -r requirements.txt
PYTHONPATH=.. uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker
```bash
docker compose up --build
```

## Forecasting methodology
The forecasting layer is designed for a realistic retail demand problem with daily demand observations by product and store. The core target is `units_sold`, and the pipeline uses only information that would have been available at prediction time.

- Baselines: naive, seasonal naive, and moving-average models.
- Feature engineering: lag features (`lag_1`, `lag_7`, `lag_14`), rolling statistics, day-of-week, month, year, promotion, holiday, price, and lagged inventory indicators.
- Validation: expanding-window time-series split using actual dates, not random train/test splitting, to avoid leakage.
- Model comparison: Random Forest, XGBoost, and LightGBM are trained and compared using MAE, RMSE, MAPE, and WMAPE.
- Production selection: the best feature-based model is chosen from validation performance and saved for inference. The current persisted artifact is LightGBM.

## API examples
```bash
curl "http://localhost:8000/api/v1/forecast?product_id=P101&store_id=1&forecast_horizon=14"
```

### What-if simulation

`POST /api/v1/simulate` runs inference with the persisted production model for a
selected product/store and scenario price, promotion, holiday, lead time,
inventory, and 1-30 day horizon. The response includes baseline and scenario
forecasts, forecast-based safety stock/reorder calculations, inventory risks,
impact deltas, and real SHAP contributions. Historical observations remain
separate from scenario values; future demand is generated recursively without
using future actual sales. Open purchase orders are not tracked.

Use `GET /api/v1/simulation/options` to populate product and store selectors.

### Model monitoring

`GET /api/v1/monitoring` evaluates the real reference/current dataset windows
against the persisted model. It reports data quality, PSI feature/target/
prediction drift, MAE/RMSE/MAPE/WMAPE comparisons, and calculated alerts. The
default reference is the earliest 80% of the available historical data and the
monitoring window is the available current data. PSI conventions are healthy
below 0.10, warning from 0.10, and critical from 0.25; these are operational
signals, not guarantees of model failure.

### Model registry and retraining

The lightweight file-based registry records the production champion and
validated/rejected challengers with model versions, training run IDs, metrics,
and artifact hashes. `GET /api/v1/models` lists registered records,
`GET /api/v1/models/production` returns the champion, and
`POST /api/v1/models/retrain` explicitly trains and evaluates a challenger.
Promotion requires a configurable 1% relative WMAPE improvement; otherwise the
champion is restored. Registered versions can be safely rolled back with
`POST /api/v1/models/{model_version}/rollback`. Retraining is never triggered
by application startup or read-only APIs.

## Validation
```bash
cd backend
PYTHONPATH=. pytest tests/test_forecast_api.py
```
