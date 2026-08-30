# Demand Intelligence: AI-Powered Demand Forecasting & Inventory Optimization Platform

This project is a production-style portfolio application for retail demand forecasting and inventory optimization. It combines data generation, feature engineering, forecasting, a FastAPI backend, and a Next.js dashboard.

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

## API examples
```bash
curl "http://localhost:8000/api/v1/forecast?product_id=P101&store_id=1&forecast_horizon=14"
```

## Validation
```bash
cd backend
PYTHONPATH=. pytest tests/test_forecast_api.py
```
