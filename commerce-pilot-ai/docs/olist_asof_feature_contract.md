# Olist As-of Feature Contract

Contract version: `olist-asof-payment-approval-v1`. A column's presence in the final export does not prove availability at payment approval. Derived features inherit the strictest classification of their inputs.

| Source | Column(s) | Meaning / earliest plausible availability | Class | Reason / order-level treatment |
|---|---|---|---|---|
| Orders | `order_id` | Order identity, creation | IDENTIFIER ONLY | Audit/join key; never predictive input. |
| Orders | `customer_id` | Customer-order identity, creation | IDENTIFIER ONLY | Join key; raw ID excluded. |
| Orders | `order_status` | Final/current lifecycle status | FORBIDDEN | Contains post-approval outcomes/cancellations. |
| Orders | `order_purchase_timestamp` | Purchase event | ALLOWED | Calendar/elapsed-to-approval derivations allowed. |
| Orders | `order_approved_at` | Confirmed prediction point | ALLOWED | Calendar and elapsed-time derivations only. |
| Orders | `order_delivered_carrier_date` | Actual later handoff | FORBIDDEN | Direct post-approval leakage. |
| Orders | `order_delivered_customer_date` | Actual outcome | TARGET CONSTRUCTION ONLY | Label only. |
| Orders | `order_estimated_delivery_date` | Recorded estimate; as-of version unproven | TARGET CONSTRUCTION ONLY | Label only. A promised-lead-time derivative is REQUIRES VERIFICATION. |
| Items | `order_id` | Order identity | IDENTIFIER ONLY | Join key. |
| Items | `order_item_id` | Within-order sequence | CONDITIONALLY ALLOWED | Count after snapshot-at-approval verification. |
| Items | `product_id`, `seller_id` | Entity identities | IDENTIFIER ONLY | Join/diversity keys; raw values excluded. |
| Items | `shipping_limit_date` | Seller shipping deadline | REQUIRES VERIFICATION | Repository does not establish when fixed/visible. |
| Items | `price`, `freight_value` | Item commercial facts | CONDITIONALLY ALLOWED | Sum/mean/max only after approval-time snapshot verification. |
| Payments | `order_id` | Order identity | IDENTIFIER ONLY | Join key. |
| Payments | `payment_sequential` | Multiple-payment sequence | CONDITIONALLY ALLOWED | Payment count only after all approval-time records are known. |
| Payments | `payment_type`, `payment_installments`, `payment_value` | Confirmed payment facts | CONDITIONALLY ALLOWED | Indicators/max/sum; verify complete at prediction event. |
| Customers | `customer_id`, `customer_unique_id` | Pseudonymous identity | IDENTIFIER ONLY | Joins/history only; no raw ID feature. |
| Customers | `customer_zip_code_prefix`, `customer_city`, `customer_state` | Customer location | CONDITIONALLY ALLOWED | Use privacy-approved granularity known at purchase. |
| Products | `product_id` | Product identity | IDENTIFIER ONLY | Join/history only. |
| Products | `product_category_name`, `product_name_lenght`, `product_description_lenght`, `product_photos_qty` | Catalog attributes | CONDITIONALLY ALLOWED | Aggregate using an approval-time catalog snapshot; spelling is source schema. |
| Products | `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm` | Physical properties | CONDITIONALLY ALLOWED | Deterministic aggregates; train-fitted missing handling. |
| Sellers | `seller_id` | Seller identity | IDENTIFIER ONLY | Join/history only. |
| Sellers | `seller_zip_code_prefix`, `seller_city`, `seller_state` | Seller location | CONDITIONALLY ALLOWED | Aggregate per order; verify approval-time seller snapshot. |
| Geolocation | `geolocation_zip_code_prefix` | Location lookup key | IDENTIFIER ONLY | Must first collapse duplicated prefixes. |
| Geolocation | `geolocation_lat`, `geolocation_lng` | Coordinates | CONDITIONALLY ALLOWED | Deterministic prefix centroid from training-approved reference only. |
| Geolocation | `geolocation_city`, `geolocation_state` | Reference geography | CONDITIONALLY ALLOWED | Deterministic mapping; ambiguity retained/reported. |
| Reviews | `review_id`, `order_id` | Review/order identifiers | FORBIDDEN | Entire review table is post-outcome. |
| Reviews | `review_score`, `review_comment_title`, `review_comment_message` | Post-delivery feedback | FORBIDDEN | Direct outcome leakage. |
| Reviews | `review_creation_date`, `review_answer_timestamp` | Post-delivery events | FORBIDDEN | Direct temporal leakage. |
| Translation | `product_category_name` | Taxonomy key | IDENTIFIER ONLY | Join key. |
| Translation | `product_category_name_english` | Static category label | CONDITIONALLY ALLOWED | Equivalent category representation after snapshot verification. |

Historical customer/seller/product performance features are CONDITIONALLY ALLOWED only when computed from outcomes strictly earlier than each order's approval time, with validation/test rows transformed using training-past information. Any target-derived statistic, validation/test outcome aggregate, or globally fit encoding is FORBIDDEN.

## Phase 1D resolution

The canonical machine-readable successor is `configs/olist_feature_contract_v1.yaml`. It narrows the primary strict core to calendar parts derived from purchase/approval timestamps and purchase-to-approval duration. Phase 1C's `ALLOWED` entries map to `STRICT_CORE_ALLOWED`; snapshot-dependent entries remain conditional; shipping-limit is conservatively forbidden; promised lead time remains unresolved. Previous evidence classifications are preserved above.
