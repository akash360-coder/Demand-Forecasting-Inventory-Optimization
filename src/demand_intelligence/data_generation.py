from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "retail_sales_sample.csv"


def generate_retail_dataset(
    start_date: str = "2023-01-01",
    periods: int = 730,
    n_products: int = 18,
    n_stores: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    categories = ["Electronics", "Home", "Apparel", "Beauty", "Sports"]
    regions = ["North", "South", "West", "East"]
    product_names = {
        0: "Laptop 15\"",
        1: "Phone Pro",
        2: "Smartwatch",
        3: "Coffee Maker",
        4: "Vacuum",
        5: "Sneakers",
        6: "Backpack",
        7: "Skin Serum",
        8: "Yoga Mat",
        9: "Treadmill",
        10: "Monitor",
        11: "USB Dock",
        12: "Blender",
        13: "Desk Lamp",
        14: "Running Top",
        15: "Hair Dryer",
        16: "Fitness Band",
        17: "Travel Case",
    }
    records: list[dict[str, object]] = []

    for product_idx in range(n_products):
        product_id = f"P{100 + product_idx}"
        category = categories[product_idx % len(categories)]
        base_demand = 35 + (product_idx % 7) * 8 + rng.integers(10, 25)
        price = 20 + rng.integers(12, 85) + (product_idx % 5) * 6
        life_cycle = 1 + 0.12 * np.sin(product_idx / 5)

        for store_idx in range(n_stores):
            store_id = store_idx + 1
            region = regions[store_idx % len(regions)]
            inventory_bias = 1.1 + rng.random() * 0.7

            for current_date in dates:
                weekday = current_date.dayofweek
                seasonal = 1 + 0.35 * np.sin((current_date.dayofyear / 365) * 2 * np.pi + product_idx / 3)
                weekend = 1.18 if weekday >= 5 else 1.0
                promotion = 1.0 if rng.random() < 0.18 else 0.0
                promotion_effect = 1.35 if promotion else 1.0
                holiday = 1.0 if current_date.month in {11, 12, 1} and (weekday in {5, 6} or current_date.day in {25, 26, 27}) else 0.0
                holiday_effect = 1.4 if holiday else 1.0
                region_factor = {"North": 1.12, "South": 0.97, "West": 1.06, "East": 1.02}[region]
                trend = life_cycle * (1 + (current_date.toordinal() - dates.min().toordinal()) / (len(dates) * 2.5))
                noise = rng.normal(0, 0.18)
                units_sold = max(
                    0,
                    round(
                        base_demand
                        * seasonal
                        * weekend
                        * promotion_effect
                        * holiday_effect
                        * region_factor
                        * trend
                        * (1 + noise)
                    ),
                )
                lead_time_days = int(rng.integers(3, 12))
                inventory_on_hand = max(20, int(units_sold * inventory_bias * (1.2 + rng.random() * 0.8)))
                records.append(
                    {
                        "date": current_date.strftime("%Y-%m-%d"),
                        "product_id": product_id,
                        "product_name": product_names.get(product_idx, f"Product {product_idx}"),
                        "store_id": store_id,
                        "region": region,
                        "category": category,
                        "units_sold": units_sold,
                        "price": round(price * (0.9 + rng.random() * 0.35), 2),
                        "promotion": int(bool(promotion)),
                        "holiday": int(bool(holiday)),
                        "inventory_on_hand": inventory_on_hand,
                        "supplier_lead_time_days": lead_time_days,
                    }
                )

    df = pd.DataFrame(records)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def ensure_dataset(path: Path | str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else DATA_PATH
    if not dataset_path.exists():
        return generate_retail_dataset()
    return pd.read_csv(dataset_path)
