import os

import pytest

os.environ.setdefault("DATABASE_MODE", "sqlite")

from app.core import config as config_module
from app.core import database as database_module
from app.models import Product, Sales
from app.services.seed_demo import seed_demo_data


def _build_settings(monkeypatch, **env):
    for key in [
        "DATABASE_URL",
        "DATABASE_MODE",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]:
        monkeypatch.delenv(key, raising=False)

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    return config_module.Settings()


def test_database_url_takes_priority(monkeypatch):
    settings = _build_settings(
        monkeypatch,
        DATABASE_URL="postgresql+psycopg2://primary:secret@prod-db:5432/app",
        DATABASE_MODE="postgresql",
        POSTGRES_USER="fallback-user",
        POSTGRES_PASSWORD="fallback-pass",
        POSTGRES_HOST="fallback-host",
        POSTGRES_PORT="5432",
        POSTGRES_DB="fallback-db",
    )
    assert settings.database_url == "postgresql+psycopg2://primary:secret@prod-db:5432/app"


def test_postgres_variables_generate_url(monkeypatch):
    settings = _build_settings(
        monkeypatch,
        POSTGRES_USER="postgres",
        POSTGRES_PASSWORD="postgres",
        POSTGRES_HOST="db",
        POSTGRES_PORT="5432",
        POSTGRES_DB="demand_intelligence",
    )
    assert settings.database_url == "postgresql+psycopg2://postgres:postgres@db:5432/demand_intelligence"


def test_missing_database_configuration_requires_explicit_sqlite(monkeypatch):
    with pytest.raises(ValueError):
        _build_settings(monkeypatch)


def test_explicit_sqlite_mode_allows_sqlite(monkeypatch):
    settings = _build_settings(monkeypatch, DATABASE_MODE="sqlite")
    assert settings.database_url == "sqlite:///./demand_intelligence.db"


def test_database_configuration_errors_are_safe(monkeypatch):
    bad_url = "postgresql+psycopg2://appuser:supersecret@127.0.0.1:1/appdb"
    with pytest.raises(database_module.DatabaseConfigurationError) as exc_info:
        database_module.configure_database(bad_url)
    message = str(exc_info.value)
    assert "supersecret" not in message
    assert "cannot be reached" in message


def test_database_seed_creates_records(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'demand_test.db'}"
    database_module.configure_database(database_url)
    database_module.init_db(database_url)
    seed_demo_data(database_url)

    with database_module.SessionLocal() as db:
        assert db.query(Product).count() > 0
        assert db.query(Sales).count() > 0
