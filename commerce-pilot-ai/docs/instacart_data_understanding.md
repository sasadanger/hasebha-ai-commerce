# Instacart Data Understanding

## Scope and files inspected

Observed facts come only from the six manually supplied Instacart CSV files. The archive origin, version, publisher checksum, and license remain Not verified.

| File | Rows | Schema and inferred types | Candidate-key result |
|---|---:|---|---|
| `aisles.csv` | 134 | `aisle_id` integer, `aisle` string | No duplicate `aisle_id`. |
| `departments.csv` | 21 | `department_id` integer, `department` string | No duplicate `department_id`. |
| `order_products__prior.csv` | 32,434,489 | order/product/cart-position/reordered integers | No duplicate `(order_id, product_id)`. |
| `order_products__train.csv` | 1,384,617 | same four integer fields | No duplicate `(order_id, product_id)`. |
| `orders.csv` | 3,421,083 | order/user/order-sequence/day/hour/interval plus `eval_set` | No duplicate `order_id`. `days_since_prior_order` has 206,209 nulls; other fields have no nulls. |
| `products.csv` | 49,688 | product ID/name/aisle/department | No duplicate `product_id`; no missing taxonomy references. |

## Observed relationships and ranges

`orders.csv` associates orders with users and order sequence. The two order-product tables associate products and cart positions with orders. Products reference aisles and departments; zero orphan product taxonomy references were observed.

- `eval_set`: prior 3,214,874; train 131,209; test 75,000.
- Order number ranges from 1 to 100; day-of-week from 0 to 6; hour parses fully as integers from 0 to 23.
- `days_since_prior_order` ranges from 0 to 30 where present. Its null count equals 206,209 and is retained because absence can be structurally meaningful for first orders.
- Prior reordered flags: 0 for 13,307,953 rows and 1 for 19,126,536 rows.

## Limitations and privacy

The files do not provide absolute order dates, prices, inventory, availability, Egyptian catalog mappings, Arabic product text, local fulfillment context, or live-store consent state. User IDs enable longitudinal behavior linkage within Instacart and require controlled access even though no direct identity fields were observed.

These historical grocery interactions cannot establish expected behavior for a future Egyptian store, other retail categories, or a different catalog.

