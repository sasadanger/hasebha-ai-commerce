# Olist Data Understanding

## Scope and files inspected

Observed facts come from the nine CSV files under `data/raw/olist/extracted/` and the reproducible Phase 1B summaries. The archive and tables were inspected only within the Olist domain.

| File | Rows | Observed columns | Inferred types and key observations |
|---|---:|---|---|
| `olist_customers_dataset.csv` | 99,441 | `customer_id`, `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state` | All strings; no nulls; `customer_id` has no duplicates. |
| `olist_geolocation_dataset.csv` | 1,000,163 | ZIP prefix, latitude, longitude, city, state fields | Coordinates are numeric; no nulls; no coordinates outside valid latitude/longitude bounds. ZIP prefixes repeat and are not a row key. |
| `olist_order_items_dataset.csv` | 112,650 | order/item/product/seller IDs, shipping limit, price, freight | IDs are string/integer; shipping limit is timestamp; values numeric; `(order_id, order_item_id)` has no duplicates. |
| `olist_order_payments_dataset.csv` | 103,886 | order, sequence, type, installments, value | No nulls observed; multiple records per order are possible by structure. |
| `olist_order_reviews_dataset.csv` | 99,224 | review/order IDs, score, title, message, creation/answer times | Timestamps inferred; 87,656 null titles and 58,247 null messages. |
| `olist_orders_dataset.csv` | 99,441 | order/customer IDs, status, purchase, approval, carrier, delivery, estimate timestamps | `order_id` has no duplicates; 160 approval, 1,783 carrier, and 2,965 customer-delivery timestamps are null. |
| `olist_products_dataset.csv` | 32,951 | product ID/category plus description, photo, weight and dimension fields | `product_id` has no duplicates; 610 nulls in category/name-length/description-length/photo count; two nulls in each physical measurement. |
| `olist_sellers_dataset.csv` | 3,095 | seller ID, ZIP prefix, city, state | No nulls; `seller_id` has no duplicates. |
| `product_category_name_translation.csv` | 71 | source and English category names | No nulls or duplicate source-category keys. |

## Observed distributions and ranges

- Purchase timestamps span `2016-09-04 21:15:19` to `2018-10-17 17:30:18`.
- Order statuses: delivered 96,478; shipped 1,107; canceled 625; unavailable 609; invoiced 314; processing 301; created 5; approved 2.
- Review scores 1–5 occur 11,424; 3,151; 8,179; 19,142; and 57,328 times respectively.
- Item prices range from 0.85 to 6,735.00; freight values range from 0.00 to 409.68. No nonpositive prices or negative freight values were observed.

## Suspicious values and limitations

No approval occurred before purchase, but 1,359 carrier timestamps precede approval timestamps and 23 customer-delivery timestamps precede carrier timestamps. These are retained and require event-semantics review. There are 7,827 records where observed customer delivery is later than the estimated date; this is an observable candidate outcome, not a business conclusion.

The data describe a historical Brazilian marketplace period, not Egypt. Geography, carriers, service levels, seller behavior, language, currency context, regulation, and operating processes may differ. Customer, seller, location, and free-text review fields require access minimization; pseudonymous IDs still support behavioral linkage within Olist and should be treated as potentially sensitive.

