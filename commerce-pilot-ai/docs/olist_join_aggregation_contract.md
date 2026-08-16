# Olist Join and Aggregation Contract

The future audit cohort must contain exactly one row per eligible `order_id`. Phase 1C does not materialize that feature matrix.

## Deterministic rules

- Start from the unique eligible orders relation.
- Items: group by `order_id` before joining. Conditional candidates are item count, distinct product count, distinct seller count, total price, total freight, maximum freight, mean freight, category diversity, and aggregate physical properties. Missing physical properties remain missing or receive train-fitted imputation later.
- Payments: group before joining. Conditional candidates are payment-record count, total payment value, maximum installments, and deterministic payment-type multi-hot indicators. An eligible order without payments receives missing indicators, not a fabricated zero payment.
- Products: join items to the unique product table before order aggregation. Unknown categories stay an explicit unknown value fitted from training rules.
- Sellers: join items to the unique seller table before order aggregation; aggregate locations/diversity.
- Category translation: many-to-one lookup after validating unique source category.
- Customers: one customer row per order through unique `customer_id`; `customer_unique_id` is not a direct feature.
- Geolocation: never join the raw one-million-row table directly. Collapse to one deterministic record per ZIP prefix first—for example count and mean latitude/longitude, with ambiguity counts—then join.
- Reviews: never join; all columns are forbidden.

Observed primary-cohort facts: 9,476 orders have multiple items, 1,223 have multiple sellers, 2,838 have multiple payment rows, zero lack items, and one lacks payments. Geolocation has 1,000,163 rows but only 19,015 distinct ZIP prefixes.

The audit proves item-aggregate and payment-aggregate joins each return 95,082 rows and 95,082 distinct orders. Candidate/forbidden column overlap is empty. Tests repeat the aggregate checks and require deterministic equality.

