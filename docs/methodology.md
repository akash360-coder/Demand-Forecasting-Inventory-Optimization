# Demand Forecasting Methodology

## Executive Summary

This document describes the professional demand forecasting pipeline implemented as part of this project. The pipeline follows industry best practices for time-series forecasting in retail inventory optimization, with strict emphasis on preventing data leakage and maintaining production-ready code standards.

---

## 1. Business Problem

**Objective:** Accurately forecast demand (units sold) for products across multiple store locations to enable optimal inventory planning and reduce stockout/overstock risks.

**Target Variable:** `units_sold` (daily demand in units per product per store)

**Use Cases:**
- Inventory replenishment planning
- Safety stock calculation
- Demand-driven procurement
- Resource allocation
- Trend analysis for business decisions

---

## 2. Dataset Overview

### Source
- **Format:** CSV (data/sample/retail_sales_sample.csv)
- **Grain:** Daily observations per product per store
- **Time Period:** 2023-01-01 to present
- **Volume:** 100,000+ rows (varies per data refresh)

### Key Columns
| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Transaction/observation date |
| `product_id` | String | Unique product identifier (e.g., P100, P101) |
| `store_id` | Integer | Unique store/location identifier |
| `units_sold` | Float | **TARGET**: Demand in units |
| `price` | Float | Unit price on the date |
| `promotion` | Float | Promotion flag/indicator (0/1) |
| `holiday` | Float | Holiday flag/indicator (0/1) |
| `inventory_on_hand` | Float | Available inventory at period start |
| `supplier_lead_time_days` | Float | Supplier lead time |
| `region` | String | Geographic region |
| `category` | String | Product category |

### Data Quality
- **Missing Values:** Minimal; forward-fill strategy applied
- **Duplicates:** Removed (key: date × product × store)
- **Invalid Records:** Negative demand, zero/negative prices removed
- **Outliers:** Identified but retained for seasonal/promotional events

---

## 3. Forecasting Target

**Variable:** `units_sold`  
**Metric:** Daily demand in units  
**Frequency:** Daily  
**Horizon:** 7, 14, 30 days (configurable)

### Target Characteristics
- Non-negative continuous variable
- Exhibits seasonal patterns (7-day and 28-day cycles detected)
- Influenced by promotions, holidays, and price changes
- Multiple hierarchical levels (product, store, region)

---

## 4. Feature Engineering

### 4.1 Date Features
Extracted from the transaction date to capture temporal patterns:

| Feature | Type | Description |
|---------|------|-------------|
| `year` | Integer | Calendar year |
| `month` | Integer | Calendar month (1-12) |
| `quarter` | Integer | Calendar quarter (1-4) |
| `week_of_year` | Integer | ISO week number |
| `day_of_week` | Integer | Day of week (0=Monday, 6=Sunday) |
| `day_of_month` | Integer | Day of month |
| `day_of_year` | Integer | Day of year (1-365/366) |
| `is_weekend` | Binary | 1 if Saturday/Sunday, 0 otherwise |

**Rationale:** Captures weekly, monthly, quarterly, and annual seasonality.

### 4.2 Lag Features
Historical demand at specified intervals, properly shifted to prevent leakage:

| Feature | Lag | Description |
|---------|-----|-------------|
| `lag_1` | 1 day | Demand from previous day |
| `lag_7` | 7 days | Demand from one week ago |
| `lag_14` | 14 days | Demand from two weeks ago |
| `lag_28` | 28 days | Demand from four weeks ago |

**Implementation:** Each lag is created using `.shift(n)` on historical data within grouped contexts (product × store). **No future values are included.**

**Rationale:** Captures short-term momentum and seasonal repetition.

### 4.3 Rolling Statistics
Summary statistics over historical windows, properly shifted:

| Feature | Window | Description |
|---------|--------|-------------|
| `rolling_mean_7` | 7 days | Average demand (past 7 days) |
| `rolling_mean_14` | 14 days | Average demand (past 14 days) |
| `rolling_mean_28` | 28 days | Average demand (past 28 days) |
| `rolling_std_7` | 7 days | Volatility (past 7 days) |
| `rolling_std_28` | 28 days | Volatility (past 28 days) |

**Implementation:** Features created as `.shift(1).rolling(window).agg()` to ensure current/future values are never included.

**Rationale:** Captures trend and volatility changes.

### 4.4 Business Features
External factors influencing demand:

| Feature | Type | Description |
|---------|------|-------------|
| `price` | Float | Unit price for the period |
| `price_change` | Float | Percent change from previous price |
| `price_index` | Float | Price relative to product average |
| `promotion` | Binary | Active promotion flag |
| `holiday` | Binary | Holiday flag |
| `inventory_on_hand_lag_1` | Float | Previous period's inventory |
| `inventory_coverage` | Float | Days of inventory on hand |
| `lead_time_days` | Float | Supplier lead time |

**Rationale:** Captures promotional effects, pricing elasticity, and supply chain constraints.

### 4.5 Feature Validation
- **Total Features:** 18 features
- **Leakage Check:** All features use only historical/lagged data
- **Availability at Prediction Time:** All features can be computed for future periods
- **Missing Value Handling:** Rows with >1 missing critical feature are dropped

---

## 5. Strict Data Leakage Prevention

### 5.1 What is Data Leakage?
Data leakage occurs when information from the future (relative to prediction time) is accidentally included in features, inflating model performance metrics.

### 5.2 Leakage Prevention Strategies

#### A. Lag Features
**Problem:** Including today's or future demand in a lag feature.  
**Solution:** Use `.shift(n)` to ensure lag_k uses data from at least k days ago.
```python
# CORRECT:
lag_1 = sales.shift(1)  # Demand from 1 day ago

# WRONG:
lag_1 = sales.iloc[-1]  # Without shift, includes current/future
```

#### B. Rolling Features
**Problem:** Rolling window includes current/future values.  
**Solution:** Shift before rolling, then only use past observations.
```python
# CORRECT:
rolling_mean_7 = sales.shift(1).rolling(7).mean()  # Past 7 days only

# WRONG:
rolling_mean_7 = sales.rolling(7).mean()  # Includes current value
```

#### C. Preprocessing/Normalization
**Problem:** Scaling parameters fitted on validation/test data.  
**Solution:** Fit preprocessing (scaler, encoder) only on training data.
```python
# CORRECT:
scaler.fit(X_train)
X_scaled = scaler.transform(X)  # Apply to all data

# WRONG:
scaler.fit(X)  # Fits on all data including test
X_scaled = scaler.transform(X)
```

#### D. Multi-Step Forecasting
**Problem:** Using actual future values to generate future features.  
**Solution:** Use predicted values recursively or NaN for unavailable data.
```python
# CORRECT (recursive):
for t in range(1, horizon+1):
    pred[t] = model.predict(features[t])
    features[t+1] = compute_features(pred[1:t])  # Uses predictions only

# WRONG:
features[t+1] = compute_features(actual[1:t+1])  # Uses actual future
```

### 5.3 Automated Leakage Detection
The codebase includes automated tests (`src/demand_intelligence/leakage_detection.py`) that verify:
- ✅ Lag features use only past data
- ✅ Rolling features don't include current/future
- ✅ Target variable is not in feature set
- ✅ Multi-step forecasting uses predicted values
- ✅ Preprocessing would be fitted on training data only

---

## 6. Baseline Models

Baseline models establish performance benchmarks against which ML models are compared.

### 6.1 Naive Forecast
**Method:** Forecast = Last observed value  
**Formula:** ŷ(t+h) = y(t)  
**Rationale:** Simple persistence assumption  
**Use Case:** Stable, trendless series  

### 6.2 Seasonal Naive Forecast
**Method:** Forecast = Value from same period last season  
**Formula:** ŷ(t+h) = y(t+h-s) where s = seasonal period  
**Rationale:** Captures weekly/seasonal repetition  
**Use Case:** Strong seasonal patterns  

### 6.3 Moving Average Forecast
**Method:** Forecast = Average of last k observations  
**Formula:** ŷ(t+h) = mean(y[t-k:t])  
**Rationale:** Smooths noise while preserving recent trends  
**Use Case:** Noisy data with moderate trends  

### Seasonal Period Detection
- Automatically detects seasonal period from data
- Tests periods: 7 (weekly), 14 (biweekly), 28 (monthly)
- Selects period with highest autocorrelation

---

## 7. Machine Learning Models

### 7.1 Random Forest
**Algorithm:** Ensemble of decision trees with bagging  
**Hyperparameters:**
- `n_estimators`: 300 trees
- `max_depth`: unlimited (grows to fit data)
- `min_samples_leaf`: 2
- `random_state`: 42 (for reproducibility)

**Strengths:** Robust, captures non-linear relationships, feature importance  
**Weaknesses:** Slower than gradient boosting, less interpretable  

### 7.2 XGBoost
**Algorithm:** Gradient boosting with sequential tree building  
**Hyperparameters:**
- `n_estimators`: 400 trees
- `max_depth`: 8
- `learning_rate`: 0.05
- `subsample`: 0.9 (90% of training data per tree)
- `colsample_bytree`: 0.9 (90% of features per tree)
- `random_state`: 42

**Strengths:** Excellent performance, regularization prevents overfitting  
**Weaknesses:** More parameters to tune  

### 7.3 LightGBM
**Algorithm:** Gradient boosting with leaf-wise growth  
**Hyperparameters:**
- `n_estimators`: 400 trees
- `learning_rate`: 0.05
- `num_leaves`: 31
- `subsample`: 0.9
- `colsample_bytree`: 0.9
- `random_state`: 42

**Strengths:** Extremely fast, handles large datasets  
**Weaknesses:** Risk of overfitting without careful regularization  

### Model Comparison
All models are trained and evaluated on identical data splits with identical features. Selection is based on validation set performance (WMAPE priority).

---

## 8. Time-Series Validation Strategy

### 8.1 Problem with Random Split
**Issue:** Randomly splitting time-series data violates the temporal order assumption. Future observations might appear in training, inflating performance metrics.

**Example:**
```
Data: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep]
Bad:  Train [Jan, Mar, May, Jul, Sep], Test [Feb, Apr, Jun, Aug]  ✗
       Model sees future during training
Good: Train [Jan, Feb, Mar, Apr, May], Validation [Jun, Jul], Test [Aug, Sep]  ✓
      Temporal order preserved
```

### 8.2 Expanding-Window (Walk-Forward) Validation
**Method:** Train on expanding historical window, validate on subsequent period

```
│ TRAIN [t₁...t_k] │ VALIDATION [t_k+1...t_m] │ TEST [t_m+1...t_end] │
│ ←─── growing ────→│ ← fixed →│ ← held-out →│
```

**Process:**
1. Split data chronologically:
   - Training: First ~70% of data
   - Validation: Next ~20% of data  
   - Test: Final ~10% of data
2. Train model on TRAIN only
3. Evaluate on VALIDATION
4. Final evaluation on TEST

**Advantages:**
- ✅ Respects temporal order
- ✅ Models never see future data
- ✅ Realistic deployment scenario
- ✅ Detects overfitting

**Implementation:** `src/demand_intelligence/forecasting.time_series_split()`

---

## 9. Evaluation Metrics

### 9.1 Mean Absolute Error (MAE)
**Formula:** MAE = (1/n) × Σ|y_actual - y_predicted|  
**Units:** Same as target (units)  
**Interpretation:** Average absolute forecast error  
**Pros:** Intuitive, same scale as demand  
**Cons:** Doesn't penalize large errors more  

### 9.2 Root Mean Squared Error (RMSE)
**Formula:** RMSE = √[(1/n) × Σ(y_actual - y_predicted)²]  
**Units:** Same as target (units)  
**Interpretation:** Quadratic average error  
**Pros:** Penalizes large errors heavily  
**Cons:** Sensitive to outliers  

### 9.3 Mean Absolute Percentage Error (MAPE)
**Formula:** MAPE = (100/n) × Σ|y_actual - y_predicted| / |y_actual|  
**Units:** Percentage (%)  
**Interpretation:** Average error as % of actual demand  
**Pros:** Scale-independent, easy to communicate  
**Cons:** Undefined when actual=0, heavily penalizes small demands  

### 9.4 Weighted Mean Absolute Percentage Error (WMAPE)
**Formula:** WMAPE = Σ|y_actual - y_predicted| / Σ|y_actual| × 100  
**Units:** Percentage (%)  
**Interpretation:** Total error as % of total demand  
**Pros:** ✅ **PRIMARY METRIC** - Business-oriented, handles zero demand  
**Cons:** None for our use case  

**Business Interpretation:**
- WMAPE 10% = Forecasts deviate by 10% from actual total demand
- Example: 1000 units actual, 900 predicted = 10% error

### 9.5 Zero-Demand Handling
When demand = 0, percentage errors (MAPE/WMAPE) could be undefined.

**Solution:** Skip zero-demand records in MAPE, but include in WMAPE.
```python
# MAPE calculation - skip zero denominators
mask = actual != 0
mape = (100/count) × mean(|actual[mask] - pred[mask]| / actual[mask])

# WMAPE calculation - zero demands contribute zero to numerator
wmape = sum(|actual - pred|) / sum(|actual|) × 100
```

---

## 10. Model Selection

### Selection Criterion
**Primary:** WMAPE on validation set  
**Secondary:** RMSE (for tie-breaking)  
**Tertiary:** Training time and model size (production considerations)

### Process
1. Run experiments on all candidate models
2. Evaluate on validation set
3. Calculate WMAPE for each model
4. Select model with lowest validation WMAPE
5. Report test set metrics for final model
6. Persist to disk

### Example Output
```
Model Comparison (Validation Set):
Model                   MAE        RMSE       MAPE       WMAPE
─────────────────────────────────────────────────────────────
Naive                   25.3       32.1       14.2%      11.5%
Seasonal Naive          22.8       29.5       12.8%      10.2%
Moving Average          24.1       31.2       13.5%      10.8%
Random Forest           18.5       24.3       10.1%      8.3%
XGBoost                 17.2       22.8       9.4%       7.8% ← SELECTED
LightGBM                17.6       23.2       9.6%       7.9%
```

---

## 11. Model Persistence

### Objectives
- Reproducibility: Re-load and re-use the exact model
- Production readiness: Serve predictions without retraining
- Auditability: Track model versions and performance

### Artifact Storage
```
models/
├── production_forecast_model.joblib    # Trained model (binary)
├── training_summary.json               # Metadata and metrics
└── README.md                           # Instructions
```

### Metadata Captured
```json
{
  "model_name": "XGBoost",
  "model_version": "20250831T190245Z",
  "trained_at": "2025-08-31T19:02:45Z",
  "dataset_rows": 15000,
  "feature_columns": [...18 features...],
  "metrics": {
    "validation": {"mae": 17.2, "rmse": 22.8, "mape": 9.4, "wmape": 7.8},
    "test": {"mae": 18.1, "rmse": 23.9, "mape": 9.8, "wmape": 8.2}
  }
}
```

### Serialization Format
- **Method:** `joblib` (Python object serialization)
- **Compatibility:** XGBoost, LightGBM, scikit-learn, standard Python objects
- **File Size:** ~50-200 MB depending on model complexity

---

## 12. Multi-Step Forecasting

### Single-Step Forecast
Generate demand for a single future period.

**Input:** Current features and historical data  
**Process:** Pass features to trained model  
**Output:** Predicted units for that period  

### Multi-Step Forecast (7/14/30 days)
Generate demand forecast for multiple future periods recursively.

**Process:**
1. Start with most recent history
2. For each future step:
   a. Generate features for that date
   b. Use predicted demand for lag/rolling features
   c. Use last known values for business features
   d. Predict demand
   e. Add prediction to history
3. Repeat until forecast horizon reached

**Lag Feature Generation for Future Periods:**
```python
# Day t (known):
lag_1[t] = demand[t-1]  # From history

# Day t+1 (future):
lag_1[t+1] = predicted_demand[t]  # From previous prediction
lag_7[t+1] = predicted_demand[t-6] OR history[t-6]  # Depends on recursion depth

# Day t+7:
lag_7[t+7] = predicted_demand[t]  # From 7 steps back in predictions
```

**Business Features for Future Periods:**
- Promotion: Last known value (assume no promotion unless specified)
- Holiday: Calendar-based (known from date)
- Price: Last known value or average (assume stable)
- Lead time: Last known value

---

## 13. Forecast Database Storage

### Table: `forecasts`
```
forecast_id (PK)
product_id (FK)
store_id (FK)
forecast_date           ← Date forecast was generated
target_date            ← Date the forecast is for
model_name             ← "XGBoost", "Random Forest", etc.
model_version          ← "20250831T190245Z"
dataset_version        ← "20250831T190245Z"
feature_list           ← JSON array of features used
predicted_demand       ← Forecasted units
lower_bound            ← 95% confidence interval lower
upper_bound            ← 95% confidence interval upper
created_at (audit)
```

### Storage Pattern
- One row per forecast point (day)
- Example: 7-day forecast for product P101, store 1 = 7 rows
- Deduplication: Replace by (product, store, target_date) to avoid duplicates

### Implementation
```python
# Insert with upsert (replace duplicates)
db.merge(Forecast(...))
db.commit()
```

---

## 14. Production API Integration

### Existing Endpoints (Phase 1)
- `GET /health` - Health check
- `GET /api/v1/forecast?product_id=P101&store_id=1&forecast_horizon=7` - Main forecast

### New Endpoints (Phase 2)
- `GET /api/v1/models/performance` - Model comparison and selected model metrics

### Backward Compatibility
All Phase 1 endpoints maintained. New features added without breaking changes.

### Response Format Enhancement
Added `wmape` to forecast summary.

```json
{
  "summary": {
    "mae": 18.1,
    "rmse": 23.9,
    "mape": 9.8,
    "wmape": 8.2
  },
  "points": [...]
}
```

---

## 15. Limitations and Future Work

### Current Limitations
1. **Prediction Intervals:** Simple ±1.96σ approach; actual confidence intervals should account for model uncertainty
2. **Hierarchical Forecasting:** Forecasts are independent per product/store; no cross-level reconciliation
3. **Calendar Features:** Basic date features; no detailed holiday/event calendar
4. **Exogenous Variables:** Limited business features; could incorporate more external data
5. **Dynamic Retraining:** Model retraining is manual; no automatic trigger for performance degradation

### Recommended Future Work (Phase 3+)
- Proper prediction intervals with quantile regression
- Hierarchical forecast reconciliation
- AutoML model selection
- Dynamic retraining with monitoring
- Integration with planning/optimization
- Demand shaping impact analysis

---

## 16. Testing and Quality Assurance

### Test Coverage
- **Data Validation:** 8 tests covering schema, missing values, duplicates, dates, demand, prices, identifiers
- **Leakage Detection:** 5 automated tests verifying no data leakage
- **Baseline Models:** 7 tests for each baseline approach
- **Feature Engineering:** Multiple feature engineering tests
- **Time-Series Validation:** Split integrity and no-overlap verification
- **Model Training:** Experiment runs and model selection
- **Persistence:** Save/load and prediction verification
- **Multi-Step Forecasting:** Horizon validation and feature generation
- **Metrics:** Metric validity and edge case handling
- **Integration:** Full pipeline execution

**Total:** 40+ test cases

### Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_forecasting_phase2.py -v

# Specific test
pytest tests/test_forecasting_phase2.py::test_no_data_leakage_comprehensive -v

# With coverage
pytest tests/ --cov=src/demand_intelligence
```

---

## 17. Reproducibility

### Training Entry Point
```bash
python -m src.training
```

### What This Does
1. Loads dataset from CSV/database
2. Validates data quality
3. Cleans invalid records
4. Runs experiments (all 6 models)
5. Selects best model based on WMAPE
6. Trains final model on all training data
7. Saves model and metadata
8. Prints comprehensive summary

### Output
- Model file: `models/production_forecast_model.joblib`
- Summary: `models/training_summary.json`
- Logs: Console output with detailed progress

### Reproducibility
- Fixed random seeds (42)
- Deterministic feature engineering
- Documented hyperparameters
- Version tracking via model_version timestamp

---

## 18. Code Quality and Standards

### Best Practices Implemented
✅ Type hints on all functions  
✅ Docstrings documenting intent  
✅ Modular architecture (forecasting, features, evaluation, validation)  
✅ Comprehensive error handling  
✅ Automated testing  
✅ Data leakage detection  
✅ Reproducible seeds  
✅ Production-ready model persistence  

### File Structure
```
src/demand_intelligence/
├── forecasting.py              # Core models and experiments
├── feature_engineering.py      # Feature creation
├── evaluation.py               # Metrics computation
├── data_validation.py         # Data quality checks
├── leakage_detection.py       # Leakage tests
└── data_generation.py         # Sample data

src/
└── training.py                # Reproducible training entry point

tests/
└── test_forecasting_phase2.py # Comprehensive test suite

backend/app/
├── api/v1/endpoints/forecast.py       # API endpoints
├── services/forecast_service.py       # Business logic
├── schemas/forecast.py                # Data models
└── models/retail_models.py            # Database models
```

---

## References

1. **Time-Series Forecasting:** Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"
2. **Data Leakage:** Kaufman et al., "Leakage in Data Mining"
3. **Retail Forecasting:** Armstrong, "Principles of Forecasting" (retail chapter)
4. **Model Comparison:** Makridakis et al., "Statistical and Machine Learning forecasting methods"

---

## Contact and Support

For questions about this methodology:
- **Implementation:** See `src/demand_intelligence/`
- **Tests:** See `tests/test_forecasting_phase2.py`
- **API Documentation:** See backend `README.md`

