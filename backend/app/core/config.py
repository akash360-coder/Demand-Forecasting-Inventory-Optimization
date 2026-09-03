from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent


class DatabaseConfigurationError(ValueError):
    """Raised when the configured database settings are incomplete or unreachable."""


class Settings:
    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Demand Intelligence API")
        self.api_prefix = os.getenv("API_PREFIX", "/api/v1")
        self.dataset_path = os.getenv(
            "DATASET_PATH",
            str(PROJECT_ROOT / "data" / "sample" / "retail_sales_sample.csv"),
        )
        self.database_mode = self._resolve_database_mode()
        self.database_url = self._resolve_database_url()
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

    @staticmethod
    def _resolve_database_mode() -> str:
        database_mode = os.getenv("DATABASE_MODE", "").strip().lower()
        if database_mode and database_mode not in {"postgresql", "sqlite"}:
            raise DatabaseConfigurationError("DATABASE_MODE must be either 'postgresql' or 'sqlite'.")
        if database_mode:
            return database_mode
        return "postgresql"

    @staticmethod
    def _resolve_database_url() -> str:
        configured_database_url = os.getenv("DATABASE_URL", "").strip()
        if configured_database_url:
            return configured_database_url

        postgres_user = os.getenv("POSTGRES_USER", "").strip()
        postgres_password = os.getenv("POSTGRES_PASSWORD", "").strip()
        postgres_host = os.getenv("POSTGRES_HOST", "").strip() or "localhost"
        postgres_port = os.getenv("POSTGRES_PORT", "").strip() or "5432"
        postgres_db = os.getenv("POSTGRES_DB", "").strip()

        if all([postgres_user, postgres_password, postgres_db]):
            return (
                f"postgresql+psycopg2://{postgres_user}:{postgres_password}@"
                f"{postgres_host}:{postgres_port}/{postgres_db}"
            )

        if os.getenv("DATABASE_MODE", "").strip().lower() == "sqlite":
            return "sqlite:///./demand_intelligence.db"

        raise DatabaseConfigurationError(
            "PostgreSQL database configuration is incomplete. "
            "Set DATABASE_URL or POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB, "
            "or set DATABASE_MODE=sqlite for explicit local development."
        )


settings = Settings()
