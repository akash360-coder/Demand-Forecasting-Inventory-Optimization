# Data Dictionary

| Field | Description | Source |
| --- | --- | --- |
| date | Transaction date | Realistic simulated retail timeline |
| product_id | Unique product ID | Synthetic product catalog |
| store_id | Store location ID | Synthetic store layout |
| region | Sales region | Derived from store geography |
| category | Product category | Derived from product mapping |
| units_sold | Daily units sold | Generated demand pattern |
| price | Product price | Simulated field |
| promotion | Promotion indicator | Simulated field |
| holiday | Holiday indicator | Simulated field |
| inventory_on_hand | Quantity available | Simulated operational field |
| supplier_lead_time_days | Lead time in days | Simulated operational field |

## Real vs simulated fields
The project uses a realistic synthetic retail dataset. The fields `date`, `product_id`, `store_id`, `region`, `category`, and `units_sold` model genuine sales behaviors. Operational variables such as `price`, `promotion`, `holiday`, `inventory_on_hand`, and `supplier_lead_time_days` are generated with business logic to create a useful inventory optimization test environment.
