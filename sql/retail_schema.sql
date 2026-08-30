CREATE TABLE IF NOT EXISTS inventory_snapshot (
  snapshot_date DATE NOT NULL,
  product_id TEXT NOT NULL,
  store_id INT NOT NULL,
  region TEXT NOT NULL,
  inventory_on_hand NUMERIC(10,2) NOT NULL,
  demand_forecast NUMERIC(10,2) NOT NULL,
  recommended_order NUMERIC(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS product_demand (
  date_value DATE NOT NULL,
  product_id TEXT NOT NULL,
  store_id INT NOT NULL,
  region TEXT NOT NULL,
  category TEXT NOT NULL,
  units_sold NUMERIC(10,2) NOT NULL,
  price NUMERIC(10,2),
  promotion INTEGER NOT NULL DEFAULT 0,
  holiday INTEGER NOT NULL DEFAULT 0
);
