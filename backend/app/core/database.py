from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import DatabaseConfigurationError, settings

Base = declarative_base()
engine = None
SessionLocal = None


def configure_database(database_url: str | None = None) -> None:
    global engine, SessionLocal
    target_url = database_url or settings.database_url

    if not target_url:
        raise DatabaseConfigurationError(
            "No database configuration was provided. Set DATABASE_URL or PostgreSQL environment variables, "
            "or set DATABASE_MODE=sqlite for explicit local development."
        )

    engine = create_engine(target_url, future=True, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - network/database-dependent path
        if "sqlite" in target_url.lower():
            raise
        raise DatabaseConfigurationError(
            "PostgreSQL database is configured but cannot be reached. "
            "Check DATABASE_URL / PostgreSQL environment variables and database availability."
        ) from exc


def init_db(database_url: str | None = None) -> None:
    configure_database(database_url)
    from app.models.retail_models import Product, Forecast, Inventory, InventoryRecommendation, ModelMetric, Sales, Store  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    if SessionLocal is None:
        configure_database()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
