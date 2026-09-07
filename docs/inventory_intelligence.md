# Inventory Intelligence

Inventory Intelligence combines demand, value, variability, and on-hand inventory into operational decisions. The implementation is exposed by `GET /api/v1/analytics/inventory-intelligence`.

## Analyses

- **ABC analysis:** Products are sorted by business value, defined as the sum of `units_sold * price`. Cumulative value contribution assigns A through 80%, B through 95%, and C to the remainder. Zero-value products are classified as C.
- **XYZ classification:** Demand variability is measured with the coefficient of variation, `standard deviation / mean demand`. X is `CV <= 0.5`, Y is `0.5 < CV <= 1.0`, and Z is `CV > 1.0`. A zero or unavailable mean is handled safely.
- **ABC-XYZ matrix:** The nine combinations AX through CZ combine value importance and demand predictability. Counts, business value, and variability are reported for each segment.
- **Inventory health score:** A bounded 0-100 score combines stockout risk, excess risk, coverage, lead time, forecast error, demand variability, and recommended-order pressure. Health bands are Excellent, Healthy, Watch, Risk, and Critical.
- **Stockout risk:** Current inventory is compared with lead-time demand, reorder point, and safety stock. The result includes a score and Low, Medium, High, or Critical level.
- **Excess inventory:** Current inventory is compared with target inventory and coverage. Excess units are `max(current inventory - target inventory, 0)` and the excess percentage is relative to target inventory when target inventory is positive.
- **Opportunity detection:** Products are ranked for urgent reorder, high stockout risk, excess inventory, forecast-accuracy issues, and high demand volatility.
- **Service-level intelligence:** For service levels 90%, 95%, 98%, and 99%, safety stock uses `z * demand standard deviation * sqrt(lead time)`. Reorder point is lead-time demand plus safety stock; target inventory and recommended order follow the service-level calculations and current inventory.
- **Risk matrix:** Product-store records are grouped into health and stockout/excess risk categories, with aggregate counts and detail rows.

## API filters and behavior

The endpoint accepts these optional query parameters: `product_id`, `store_id`, `category`, `region`, `abc_class`, `xyz_class`, `risk_level`, `health_band`, `service_level`, `grouping`, `start_date`, and `end_date`.

`abc_class` must be A, B, or C; `xyz_class` must be X, Y, or Z; and `risk_level` must be Low, Medium, High, or Critical. `service_level` must be greater than 0 and less than 1. Invalid enum values return HTTP 400, and invalid numeric query values such as `service_level=1.5` return HTTP 422 through FastAPI validation.

A valid filter with no matching records returns HTTP 200 and the normal response schema with empty collections and zero summary values. A valid filter with matching records returns HTTP 200 with analytics calculated from the filtered records. A nonexistent product therefore follows the empty-result behavior rather than being treated as an invalid enum value.

## Assumptions and limitations

- The source data contains daily sales, price, inventory, and supplier lead-time values suitable for the calculations above.
- Business value uses historical sales and price; it is not a margin or profit measure.
- The implementation uses the available historical demand distribution and does not model promotions, supply disruptions, substitutions, or intermittent-demand distributions beyond the supplied features.
- Risk thresholds and service-level z-scores are operational heuristics and should be calibrated with business owners and observed service performance.
- Results are descriptive decision support, not automatic purchase orders. Data quality, date coverage, lead-time accuracy, and current inventory accuracy directly affect the recommendations.
