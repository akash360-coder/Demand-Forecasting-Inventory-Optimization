from app.core import database as database_module
from app.models import Product, Sales
from app.services.seed_demo import seed_demo_data


def test_database_seed_creates_records(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'demand_test.db'}"
    database_module.configure_database(database_url)
    database_module.init_db(database_url)
    seed_demo_data(database_url)

    with database_module.SessionLocal() as db:
        assert db.query(Product).count() > 0
        assert db.query(Sales).count() > 0
