# PRODUCTION MODEL FEATURE INVENTORY
Availability classes (RULE 8): **A** available now in HASEBHA · **B** derivable now from existing HASEBHA data · **C** available after a small engineering addition · **D** requires a business decision · **E** requires historical data that does not exist yet · **F** Olist/research-only.

All features evaluated at T0 = `order.placed` (order purchase timestamp). "Known at T0" = yes for all rows below unless noted.

## 1. Current MODEL P features (13, in production shadow model)

| Feature | Source | Class | In Olist | Univariate AUC (this study) | Mechanism |
|---|---|---|---|---|---|
| purchase_weekday / purchase_hour / purchase_month | order.created_at | A | yes | 0.519 / 0.524 / 0.501 | operational timing workload |
| same_state (served as `same_zone`, always false today) | shipping address vs store region | A (degenerate) | yes | 0.502 | route complexity |
| n_items, n_distinct_products, n_categories | order items | A | yes | 0.512 / 0.503 / 0.504 | basket complexity |
| total_price, total_freight, total_freight_over_price | items/pricing | A | yes | 0.537 / 0.531 / 0.515 | order value/freight burden |
| weight_g, volume_cm3 | product dims | C (requires product metadata; often missing in Medusa) | yes | 0.540 / 0.540 | physical handling effort |
| payment_value | payment capture | C (payment provider payload) | yes | 0.539 | value proxy |

## 2. Research (V3) features excluded from P — with measured contribution

| Feature | Class | Univariate AUC | Measured ΔAUC when removed from FULL-23 |
|---|---|---|---|
| seller_past_breach_rate_expanding | E (needs first-party history) | 0.729 | seller-history block (8 feats) together: **−0.202** |
| seller_breach_rate_30d / 90d | E | 0.730 / 0.733 | (same block) |
| seller_handling_mean_30d | E | 0.731 | (same block) |
| seller_past_handling_median/std_expanding | E | 0.714 / 0.654 | (same block) |
| seller_past_order_count, seller_recent_load_7d | E | 0.531 / 0.504 | (same block) |
| days_to_shipping_deadline | **D** (business SLA: `promise_business_days`) | 0.517 | −0.013 (interactional value) |
| n_installments | F on Olist / C via payment provider | 0.506 | −0.002 |

## 3. New candidate features tested in this investigation (Phase 5)

| Feature | Derivation path (HASEBHA) | Class | Known at T0 | Leakage risk | Result |
|---|---|---|---|---|---|
| geo_dist_km (haversine, zip centroid) | shipping address ↔ store address geocode | C | yes | none | **no signal** (Δ −0.001) |
| geo_lat_diff, geo_lng_diff | same | C | yes | none | no signal |
| cust_prior_orders, cust_is_repeat, cust_tenure_days | customer table + past orders | B | yes | none (cumcount by T0) | **no signal** (Δ +0.002) |
| cust_prior_late_rate, cust_prior_observed_orders | prior orders with outcome *delivered before T0* (strict) | E (no history yet) | yes | audited strict: outcome-date < T0 | no signal |
| product_category_code | product → category field | B | yes | none (ordinal code, no target encoding) | **+0.014, unstable** |
| item_price_mean, item_price_max, freight_per_item | order items | A | yes | none | no signal |
| store_breach_rate_30d/90d/expanding, store backlog, store load | rolling store aggregates (MODEL P+, earlier test) | E | yes | audited (shift(1)) | **negative (−0.009), REJECTED** |

## 4. Features considered and rejected without experiment

- Estimated delivery date: not present in HASEBHA (D, same SLA decision).
- Inventory/stock location, warehouse, pickup: single vendor, no warehouse system (would be C but no data source exists).
- Reviews/NLP sentiment (exists in repo as separate track): post-delivery, **leakage by construction** at T0.
- Fulfillment status at prediction time: degenerate at order.placed (always "pending").

## 5. Bottom line

Every class-A/B/C feature with a plausible mechanism has now been tested. The only untapped
legitimate gains are **product category (+0.014, marginal)** and **days_to_shipping_deadline
(+0.013, gated on a business SLA decision)**. Everything above 0.60 AUC requires class-E data:
first-party fulfillment outcomes accumulated through the shadow loop (~4,500 orders / ~410
breach events recommended).
