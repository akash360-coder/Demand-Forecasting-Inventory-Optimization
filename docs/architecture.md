# Architecture

```mermaid
flowchart LR
  A[CSV / Synthetic Retail Data] --> B[Data Validation & Feature Engineering]
  B --> C[Forecasting & Inventory Models]
  C --> D[FastAPI Service]
  D --> E[Next.js Dashboard]
  D --> F[PostgreSQL Data Store]
  C --> G[Model Card & Evaluation Reports]
```

## Components
- `src/demand_intelligence` contains the data generation, feature engineering, forecasting, and inventory logic.
- `backend/app` holds the API service and schemas.
- `frontend` renders the business dashboard in real time.
- `data/sample` stores the generated retail dataset used for the proof of concept.

## Database runtime
PostgreSQL is the recommended runtime database for Docker and deployment. SQLite remains available only when the environment explicitly sets `DATABASE_MODE=sqlite` for local development. The service does not silently default to SQLite when PostgreSQL configuration is expected but missing or incomplete.
