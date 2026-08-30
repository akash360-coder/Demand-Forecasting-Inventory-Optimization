from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

Base = declarative_base()
engine = None
SessionLocal = None


def configure_database(database_url: str | None = None) -> None:
    global engine, SessionLocal
    target_url = database_url or settings.database_url
    engine = create_engine(target_url, future=True, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


configure_database()


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
