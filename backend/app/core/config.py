from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Demand Intelligence API")
        self.api_prefix = os.getenv("API_PREFIX", "/api/v1")
        self.dataset_path = os.getenv(
            "DATASET_PATH",
            str(PROJECT_ROOT / "data" / "sample" / "retail_sales_sample.csv"),
        )
        self.database_url = self._resolve_database_url()
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

    @staticmethod
    def _resolve_database_url() -> str:
        configured_database_url = os.getenv("DATABASE_URL")
        if configured_database_url:
            return configured_database_url

        postgres_user = os.getenv("POSTGRES_USER")
        postgres_password = os.getenv("POSTGRES_PASSWORD")
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB")

        if all([postgres_user, postgres_password, postgres_db]):
            return (
                f"postgresql+psycopg2://{postgres_user}:{postgres_password}@"
                f"{postgres_host}:{postgres_port}/{postgres_db}"
            )

        return "sqlite:///./demand_intelligence.db"


settings = Settings()
