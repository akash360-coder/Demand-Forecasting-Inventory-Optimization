from __future__ import annotations

from datetime import datetime

from app.core import database as database_module
from app.models import Inventory, Product, Sales, Store
from src.demand_intelligence.data_generation import ensure_dataset


def seed_demo_data(database_url: str | None = None) -> None:
    database_module.init_db(database_url)
    db = database_module.SessionLocal()
    try:
        df = ensure_dataset()
        existing_sales = db.query(Sales).count() > 0
        products = {}
        if not existing_sales:
            for product_id, product_df in df.groupby("product_id"):
                products[product_id] = Product(
                    product_id=product_id,
                    name=product_df["product_name"].iloc[0],
                    category=product_df["category"].iloc[0],
                    active=True,
                )
            db.add_all(products.values())

        stores = {}
        if not existing_sales:
            for store_id, store_df in df.groupby("store_id"):
                stores[store_id] = Store(
                    store_id=int(store_id),
                    name=f"Store {int(store_id)}",
                    region=store_df["region"].iloc[0],
                    city=f"City {int(store_id)}",
                    active=True,
                )
            db.add_all(stores.values())
        db.flush()

        if not existing_sales:
            for record in df.to_dict(orient="records"):
                db.add(
                    Sales(
                        date=datetime.strptime(record["date"], "%Y-%m-%d").date(),
                        product_id=record["product_id"],
                        store_id=int(record["store_id"]),
                        units_sold=float(record["units_sold"]),
                        price=float(record["price"]),
                        promotion=float(record["promotion"]),
                        holiday=float(record["holiday"]),
                    )
                )

        existing_inventory = {
            (item.product_id, item.store_id)
            for item in db.query(Inventory.product_id, Inventory.store_id).all()
        }
        for (product_id, store_id), group in df.groupby(["product_id", "store_id"]):
            if (product_id, int(store_id)) not in existing_inventory:
                latest = group.sort_values("date").iloc[-1]
                db.add(
                    Inventory(
                        product_id=product_id,
                        store_id=int(store_id),
                        snapshot_date=datetime.strptime(str(latest["date"]), "%Y-%m-%d").date(),
                        inventory_on_hand=float(latest["inventory_on_hand"]),
                        lead_time_days=float(latest["supplier_lead_time_days"]),
                        safety_stock=0.0,
                        reorder_point=0.0,
                        target_inventory=0.0,
                        abc_class="C",
                    )
                )

        db.commit()
    finally:
        db.close()
