# Olist Phase 2C AS-OF Feature Audit

Status: `PLANNING_AND_AUDIT_OUTPUT`. Authorization scope: `ASOF_AUDIT_AND_FEATURE_CONTRACT_HARDENING_ONLY`. Created: 2026-08-09T00:00:00+00:00.

Decision timestamp: `order_approved_at`. Target: `late_delivery = order_delivered_customer_date > order_estimated_delivery_date`.

## Executive summary

| Classification | Count |
|---|---:|
| `PROVEN_PRIMARY_ASOF` | 0 |
| `UNVERIFIED_ASOF` | 15 |
| `RETROSPECTIVE_ONLY` | 0 |
| `FORBIDDEN_LEAKAGE` | 0 |
| `IDENTIFIER_ONLY` | 0 |
| `POST_OUTCOME` | 0 |
| `UNKNOWN` | 0 |

0 of 15 Expanded features meet the DIRECT_PROVEN evidence standard required for PROVEN_PRIMARY_ASOF (Step 6 of the governing session prompt: all 12 sub-questions must be answerable; for every one of the 15 features, at least questions 3 (timestamp <= order_approved_at), 8 (backfill), and 9 (revision process) cannot be answered from repository or source-schema evidence). All 15 remain UNVERIFIED_ASOF, unchanged from the Phase 2B closure state. Item/freight-derived features (7 of 15) carry STRONG_INFERRED evidence given the enforced purchase<=approval temporal ordering; payment-derived features (3 of 15) and product-catalog-derived features (5 of 15) carry only WEAK_INFERRED evidence because their source tables contain no timestamp or versioning information whatsoever. No feature is reclassified FORBIDDEN_LEAKAGE, POST_OUTCOME, or CONTRADICTORY; the audit finds an evidence gap, not proof of leakage. Feature-contract hardening therefore adds evidence-strength granularity and a documented reason per feature, without promoting any feature and without narrowing the existing UNVERIFIED_ASOF classification inherited from Phase 2B.

### Join-time audit

All 15 features are aggregated via GROUP BY order_id (payments, items) or product_id-join-then-group-by-order_id (products), per docs/olist_join_aggregation_contract.md and confirmed directly in src/modeling/olist/expanded_feature_builder.py. No join carries a temporal predicate because neither the payments table nor the products table contains any timestamp column (independently confirmed by parquet schema inspection this session). No cross-order fan-out or future-order-row inclusion is structurally possible, because every join is scoped to the specific order_id already restricted to the eligible/temporal cohort in the enclosing query. The unresolved risk is entirely same-order snapshot completeness/immutability (are these tables' values as exported today identical to what existed at the historical order_approved_at?), not cross-order leakage.

### Historical-aggregation audit (Step 7)

Zero of the 15 Expanded features are customer/seller/product BEHAVIORAL HISTORY aggregates (no rolling statistics, historical rates, cumulative counts, or prior-order aggregates are computed anywhere in expanded_feature_builder.py). All 15 are same-order, same-purchase-event aggregates (item/payment rows belonging to the single order being scored) or a same-order product-catalog lookup. The Step 7 concern (global groupby leakage across orders, future rows in rolling statistics, post-order outcomes in entity histories) is therefore evaluated and found NOT APPLICABLE to this specific 15-feature set; no such cross-order historical feature exists in the current Expanded contract to audit.

### Backfill/revision risk

Cannot be resolved from repository evidence for any of the 15 features. Neither olist_order_payments_dataset nor olist_products_dataset contains any timestamp or version column (confirmed by direct schema inspection), so no backfill/revision behavior can be measured, only assumed. This is the single largest evidence gap blocking PROVEN_PRIMARY_ASOF for any feature.

## Per-feature detail

### `payment_record_count`

- **What it means:** count of payment_sequential rows per order_id
- **How it is computed:** count(*) GROUP BY order_id
- **Source:** olist_order_payments_dataset
- **Relevant timestamp:** source event timestamp = none (no timestamp column in source table); latest safe cutoff = unknown (no source timestamp exists)
- **Evidence found:** olist_order_payments_dataset schema is exactly [order_id, payment_sequential, payment_type, payment_installments, payment_value] -- independently confirmed by direct parquet schema inspection this session. No event timestamp of any kind exists in this table. The build_development() SQL (expanded_feature_builder.py) aggregates ALL payment rows for the order_id with no temporal predicate, because none is possible. order_approved_at is documented as 'confirmed payment approval', which plausibly implies all payment_sequential rows for an order are registered together at authorization even when payment_installments>1 (a typical Brazilian installment-card pattern where the total is authorized upfront and charged over future months). This is a plausible domain inference, not a proven fact: the repository contains no evidence ruling out payment records being appended, corrected, or only fully known after order_approved_at.
- **Leakage/availability concern:** MEDIUM_HIGH_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_payments_dataset.parquet (schema inspection)`

### `max_payment_installments`

- **What it means:** maximum payment_installments value across the order's payment rows
- **How it is computed:** max(payment_installments) GROUP BY order_id
- **Source:** olist_order_payments_dataset
- **Relevant timestamp:** source event timestamp = none (no timestamp column in source table); latest safe cutoff = unknown (no source timestamp exists)
- **Evidence found:** olist_order_payments_dataset schema is exactly [order_id, payment_sequential, payment_type, payment_installments, payment_value] -- independently confirmed by direct parquet schema inspection this session. No event timestamp of any kind exists in this table. The build_development() SQL (expanded_feature_builder.py) aggregates ALL payment rows for the order_id with no temporal predicate, because none is possible. order_approved_at is documented as 'confirmed payment approval', which plausibly implies all payment_sequential rows for an order are registered together at authorization even when payment_installments>1 (a typical Brazilian installment-card pattern where the total is authorized upfront and charged over future months). This is a plausible domain inference, not a proven fact: the repository contains no evidence ruling out payment records being appended, corrected, or only fully known after order_approved_at.
- **Leakage/availability concern:** MEDIUM_HIGH_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_payments_dataset.parquet (schema inspection)`

### `total_payment_value`

- **What it means:** sum of payment_value across the order's payment rows
- **How it is computed:** sum(payment_value) GROUP BY order_id
- **Source:** olist_order_payments_dataset
- **Relevant timestamp:** source event timestamp = none (no timestamp column in source table); latest safe cutoff = unknown (no source timestamp exists)
- **Evidence found:** olist_order_payments_dataset schema is exactly [order_id, payment_sequential, payment_type, payment_installments, payment_value] -- independently confirmed by direct parquet schema inspection this session. No event timestamp of any kind exists in this table. The build_development() SQL (expanded_feature_builder.py) aggregates ALL payment rows for the order_id with no temporal predicate, because none is possible. order_approved_at is documented as 'confirmed payment approval', which plausibly implies all payment_sequential rows for an order are registered together at authorization even when payment_installments>1 (a typical Brazilian installment-card pattern where the total is authorized upfront and charged over future months). This is a plausible domain inference, not a proven fact: the repository contains no evidence ruling out payment records being appended, corrected, or only fully known after order_approved_at.
- **Leakage/availability concern:** MEDIUM_HIGH_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_payments_dataset.parquet (schema inspection)`

### `item_count`

- **What it means:** number of item rows for the order
- **How it is computed:** count(*) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `unique_product_count`

- **What it means:** distinct product_id count within the order
- **How it is computed:** count(distinct product_id) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `unique_seller_count`

- **What it means:** distinct seller_id count within the order
- **How it is computed:** count(distinct seller_id) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `total_item_price`

- **What it means:** sum of item price across the order
- **How it is computed:** sum(price) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `total_freight_value`

- **What it means:** sum of freight_value across the order
- **How it is computed:** sum(freight_value) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `mean_freight_value`

- **What it means:** mean of freight_value across the order's items
- **How it is computed:** avg(freight_value) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `max_freight_value`

- **What it means:** maximum freight_value across the order's items
- **How it is computed:** max(freight_value) GROUP BY order_id
- **Source:** olist_order_items_dataset
- **Relevant timestamp:** source event timestamp = implicitly order_purchase_timestamp (no direct per-item timestamp); latest safe cutoff = order_approved_at (unverified, inferred only)
- **Evidence found:** olist_order_items_dataset rows are tied 1:1 to the specific order being scored (grouped by order_id before use, per docs/olist_join_aggregation_contract.md), not aggregated across other orders, so this is not a cross-order historical-aggregation leakage pattern (Step 7 audit: not applicable to this feature). build_development() enforces order_approved_at>=order_purchase_timestamp for every eligible row, and items are the composition of what was purchased, so item-level facts (which product, which seller, price, freight) plausibly existed no later than order_purchase_timestamp, which by construction precedes order_approved_at. However, no per-item creation or modification timestamp exists in the source schema to directly prove items/prices were immutable between purchase and approval (e.g. that no item could be added, removed, or re-priced during that interval).
- **Leakage/availability concern:** LOW_MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `STRONG_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `src/modeling/olist/strict_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_order_items_dataset.parquet (schema inspection)`

### `product_category_diversity`

- **What it means:** distinct product_category_name count among the order's items
- **How it is computed:** count(distinct product_category_name) GROUP BY order_id, via items JOIN products USING(product_id)
- **Source:** olist_products_dataset (joined via items)
- **Relevant timestamp:** source event timestamp = none (no timestamp/version column in source table); latest safe cutoff = unknown (static catalog export, no versioning)
- **Evidence found:** olist_products_dataset schema is exactly [product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm] -- independently confirmed by direct parquet schema inspection this session. There is no timestamp or version field of any kind. This is a single static current-day catalog export joined by product_id with no snapshot/versioning mechanism (docs/olist_asof_feature_contract.md already flags this CONDITIONALLY_ALLOWED pending 'approval-time catalog snapshot' verification; the code confirms no such snapshot mechanism exists). A product's recorded category or physical dimensions could have been corrected by a seller/catalog admin after any given historical order was placed, and the feature would silently reflect the corrected/current value with no way to detect this from the data as exported.
- **Leakage/availability concern:** MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_products_dataset.parquet (schema inspection)`

### `mean_product_weight_g`

- **What it means:** mean product_weight_g among the order's items' products
- **How it is computed:** avg(product_weight_g) GROUP BY order_id, via items JOIN products USING(product_id)
- **Source:** olist_products_dataset (joined via items)
- **Relevant timestamp:** source event timestamp = none (no timestamp/version column in source table); latest safe cutoff = unknown (static catalog export, no versioning)
- **Evidence found:** olist_products_dataset schema is exactly [product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm] -- independently confirmed by direct parquet schema inspection this session. There is no timestamp or version field of any kind. This is a single static current-day catalog export joined by product_id with no snapshot/versioning mechanism (docs/olist_asof_feature_contract.md already flags this CONDITIONALLY_ALLOWED pending 'approval-time catalog snapshot' verification; the code confirms no such snapshot mechanism exists). A product's recorded category or physical dimensions could have been corrected by a seller/catalog admin after any given historical order was placed, and the feature would silently reflect the corrected/current value with no way to detect this from the data as exported.
- **Leakage/availability concern:** MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_products_dataset.parquet (schema inspection)`

### `mean_product_length_cm`

- **What it means:** mean product_length_cm among the order's items' products
- **How it is computed:** avg(product_length_cm) GROUP BY order_id, via items JOIN products USING(product_id)
- **Source:** olist_products_dataset (joined via items)
- **Relevant timestamp:** source event timestamp = none (no timestamp/version column in source table); latest safe cutoff = unknown (static catalog export, no versioning)
- **Evidence found:** olist_products_dataset schema is exactly [product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm] -- independently confirmed by direct parquet schema inspection this session. There is no timestamp or version field of any kind. This is a single static current-day catalog export joined by product_id with no snapshot/versioning mechanism (docs/olist_asof_feature_contract.md already flags this CONDITIONALLY_ALLOWED pending 'approval-time catalog snapshot' verification; the code confirms no such snapshot mechanism exists). A product's recorded category or physical dimensions could have been corrected by a seller/catalog admin after any given historical order was placed, and the feature would silently reflect the corrected/current value with no way to detect this from the data as exported.
- **Leakage/availability concern:** MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_products_dataset.parquet (schema inspection)`

### `mean_product_height_cm`

- **What it means:** mean product_height_cm among the order's items' products
- **How it is computed:** avg(product_height_cm) GROUP BY order_id, via items JOIN products USING(product_id)
- **Source:** olist_products_dataset (joined via items)
- **Relevant timestamp:** source event timestamp = none (no timestamp/version column in source table); latest safe cutoff = unknown (static catalog export, no versioning)
- **Evidence found:** olist_products_dataset schema is exactly [product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm] -- independently confirmed by direct parquet schema inspection this session. There is no timestamp or version field of any kind. This is a single static current-day catalog export joined by product_id with no snapshot/versioning mechanism (docs/olist_asof_feature_contract.md already flags this CONDITIONALLY_ALLOWED pending 'approval-time catalog snapshot' verification; the code confirms no such snapshot mechanism exists). A product's recorded category or physical dimensions could have been corrected by a seller/catalog admin after any given historical order was placed, and the feature would silently reflect the corrected/current value with no way to detect this from the data as exported.
- **Leakage/availability concern:** MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_products_dataset.parquet (schema inspection)`

### `mean_product_width_cm`

- **What it means:** mean product_width_cm among the order's items' products
- **How it is computed:** avg(product_width_cm) GROUP BY order_id, via items JOIN products USING(product_id)
- **Source:** olist_products_dataset (joined via items)
- **Relevant timestamp:** source event timestamp = none (no timestamp/version column in source table); latest safe cutoff = unknown (static catalog export, no versioning)
- **Evidence found:** olist_products_dataset schema is exactly [product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm] -- independently confirmed by direct parquet schema inspection this session. There is no timestamp or version field of any kind. This is a single static current-day catalog export joined by product_id with no snapshot/versioning mechanism (docs/olist_asof_feature_contract.md already flags this CONDITIONALLY_ALLOWED pending 'approval-time catalog snapshot' verification; the code confirms no such snapshot mechanism exists). A product's recorded category or physical dimensions could have been corrected by a seller/catalog admin after any given historical order was placed, and the feature would silently reflect the corrected/current value with no way to detect this from the data as exported.
- **Leakage/availability concern:** MEDIUM_UNPROVEN (join temporal constraint: none present in source code (src/modeling/olist/expanded_feature_builder.py); no temporal predicate on payments or items/products joins)
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- **Evidence strength:** `WEAK_INFERRED`
- **Why:** insufficient evidence to answer all AS-OF proof-standard questions (source timestamp existence, backfill behavior, revision process) required for `PROVEN_PRIMARY_ASOF`; classification remains unchanged from the inherited Phase 2B `UNVERIFIED_ASOF` state.
- **Evidence paths:** `docs/olist_data_understanding.md`, `docs/olist_asof_feature_contract.md`, `docs/olist_join_aggregation_contract.md`, `docs/olist_expanded_sensitivity_contract.md`, `src/modeling/olist/expanded_feature_builder.py`, `reports/generated/olist/phase2b/correction_v1/complete_fifteen_feature_audit.json`, `data/processed/olist/olist_products_dataset.parquet (schema inspection)`
