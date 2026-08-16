# Olist Phase 2C AS-OF Evidence Remediation

Status: authorization scope `ASOF_EVIDENCE_REMEDIATION_ONLY`. Created: 2026-08-09T00:00:00+00:00.

Decision timestamp: `order_approved_at`. Target: `late_delivery = order_delivered_customer_date > order_estimated_delivery_date`.

This document supersedes nothing. `reports/generated/olist/phase2c/asof_feature_audit.json`, `configs/olist_phase2c_asof_feature_contract.yaml`, and `docs/olist_phase2c_asof_feature_audit.md` (Execution Session 1) remain authoritative and unchanged, since no classification changed as a result of this remediation attempt.

## Outcome

**`ASOF_EVIDENCE_REMEDIATION = IRREDUCIBLE_IN_OLIST`**

The Olist public dataset, as released, structurally lacks the timestamp/versioning/provenance evidence needed to prove PROVEN_PRIMARY_ASOF for any of the 15 Expanded features under the order_approved_at decision timestamp. This is a dataset-governance limitation, not a model, code, or Phase 2B failure, and not a reason to fabricate timing assumptions. All 15 features remain UNVERIFIED_ASOF. No further Olist-specific evidence search is justified for any feature group -- every realistic source (official page, Olist-adjacent notebook, bundled archive documentation, academic literature, internal provenance records) was attempted or checked and either returned no usable content or no relevant information.

## Evidence hierarchy used

- **LEVEL_1:** Official dataset/source documentation
- **LEVEL_2:** Original publisher-maintained repository or schema documentation
- **LEVEL_3:** Peer-reviewed publication describing exact dataset-generation semantics
- **LEVEL_4:** Strong independently corroborated technical documentation
- **LEVEL_5:** Community explanations, tutorials, notebooks, forum posts
- **Promotion rule:** Only LEVEL 1-3 should normally be sufficient to promote a feature to PROVEN_PRIMARY_ASOF. LEVEL 4 may support a conclusion but must not alone create proof. LEVEL 5 aids discovery only.

## Research summary

A comprehensive primary-source investigation was conducted: 5 targeted WebSearch queries and 4 WebFetch attempts against the official Kaggle dataset page (2 URL variants) and an Olist-adjacent Kaggle notebook. Every WebFetch attempt against Kaggle returned only the page title -- Kaggle's dataset and notebook pages are JavaScript-rendered single-page applications that available tooling cannot render past the title, a hard technical barrier independent of query wording. No academic data-descriptor publication exists for this dataset (confirmed by dedicated search). The raw downloaded archive (data/raw/olist/extracted/) was checked directly and contains 9 CSV files with no bundled documentation. This repository's own provenance records (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema-timing detail beyond what Execution Session 1 already established. Multiple independent secondary sources (LEVEL 4-5) were found and are logged, providing corroborating-but-not-primary evidence for payment record semantics only; zero evidence of any level was found for product-catalog snapshot versioning.

## Source-system vs dataset-snapshot distinction

All available evidence, old and new, describes DATASET_SNAPSHOT_CONTENT (what the final Kaggle export contains) rather than SOURCE_SYSTEM_AVAILABILITY (what was observable in Olist's live operational system at the historical order_approved_at moment). This distinction remains unresolved for all 15 features.

## Per-table evidence results

- **Payments table:** Corroborating LEVEL 4-5 evidence found narrowing the interpretation of payment_sequential/payment_value/order_approved_at semantics; does not meet DIRECT_PROVEN standard; revision/backfill risk unaddressed by any source.
- **Order items table:** No new evidence found or realistically obtainable; unchanged STRONG_INFERRED rating from Execution Session 1's direct code/schema inspection.
- **Products table:** No evidence of any level found addressing snapshot/versioning semantics, despite comprehensive search.
- **Revision/backfill (all tables):** Unresolved for all 15 features; no source of any level addresses backfill or revision behavior for payments or products tables.

## Classification counts

| Classification | Prior (Execution Session 1) | Final (this remediation) |
|---|---:|---:|
| `PROVEN_PRIMARY_ASOF` | 0 | 0 |
| `UNVERIFIED_ASOF` | 15 | 15 |
| `RETROSPECTIVE_ONLY` | 0 | 0 |
| `FORBIDDEN_LEAKAGE` | 0 | 0 |
| `IDENTIFIER_ONLY` | 0 | 0 |
| `POST_OUTCOME` | 0 | 0 |
| `UNKNOWN` | 0 | 0 |

Features promoted: **0**. Features remaining unresolved: **15** (all 15).

## Per-feature remediation detail

### `payment_record_count`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: True — source: Multiple convergent secondary sources (Kaggle community notebooks/tutorials paraphrasing the official Kaggle data dictionary); official Kaggle page and Olist-authored notebook could not be directly rendered/quoted by available tooling
- Evidence level: `LEVEL_4_5_CORROBORATING_NOT_PRIMARY`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED_WITH_CORROBORATION`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: Corroborating secondary evidence narrows the plausible interpretation of payment_sequential/payment_value timing but does not meet the DIRECT_PROVEN standard: no LEVEL 1-3 primary source was obtainable, and revision/backfill behavior remains completely unaddressed by any source found. Promotion standard requires ALL 8 components resolved; at least 2 remain unresolved. Multiple independent secondary sources (Kaggle community notebooks, tutorial articles) consistently paraphrase/quote wording matching Olist's known public data-dictionary language: 'payment_sequential: a customer may pay an order with more than one payment method (e.g. voucher + credit card), so a given order_id will have multiple rows for the different payment methods used in one checkout' and 'payment_value: the value of the transaction.' Independently, multiple sources consistently describe order_approved_at as 'the payment approval timestamp.' Convergent secondary corroboration (LEVEL 4-5, not independently verified against the primary Kaggle page because it could not be rendered by available tooling) suggests payment_sequential's multiplicity reflects split payment METHODS agreed together at one checkout event, not sequential future installment-charge records appended over time. This somewhat narrows -- but does not resolve -- the availability-timing concern for payment_record_count and total_payment_value: even if the full agreed value/method-split is fixed at the approval event, no evidence of any level proves the payment rows are free of later correction, refund adjustment, or backfill (Step 6 requirement 6 remains unresolved), and no directly-quotable LEVEL 1-3 primary source could be obtained to elevate this beyond corroborating inference.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible to available tooling (JS-rendered SPA); no bundled documentation exists in the raw archive; no academic data descriptor exists. No further realistic named source remains.

### `max_payment_installments`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: True — source: Multiple convergent secondary sources (Kaggle community notebooks/tutorials paraphrasing the official Kaggle data dictionary); official Kaggle page and Olist-authored notebook could not be directly rendered/quoted by available tooling
- Evidence level: `LEVEL_4_5_CORROBORATING_NOT_PRIMARY`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED_WITH_CORROBORATION`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: Corroborating secondary evidence narrows the plausible interpretation of payment_sequential/payment_value timing but does not meet the DIRECT_PROVEN standard: no LEVEL 1-3 primary source was obtainable, and revision/backfill behavior remains completely unaddressed by any source found. Promotion standard requires ALL 8 components resolved; at least 2 remain unresolved. Multiple independent secondary sources (Kaggle community notebooks, tutorial articles) consistently paraphrase/quote wording matching Olist's known public data-dictionary language: 'payment_sequential: a customer may pay an order with more than one payment method (e.g. voucher + credit card), so a given order_id will have multiple rows for the different payment methods used in one checkout' and 'payment_value: the value of the transaction.' Independently, multiple sources consistently describe order_approved_at as 'the payment approval timestamp.' Convergent secondary corroboration (LEVEL 4-5, not independently verified against the primary Kaggle page because it could not be rendered by available tooling) suggests payment_sequential's multiplicity reflects split payment METHODS agreed together at one checkout event, not sequential future installment-charge records appended over time. This somewhat narrows -- but does not resolve -- the availability-timing concern for payment_record_count and total_payment_value: even if the full agreed value/method-split is fixed at the approval event, no evidence of any level proves the payment rows are free of later correction, refund adjustment, or backfill (Step 6 requirement 6 remains unresolved), and no directly-quotable LEVEL 1-3 primary source could be obtained to elevate this beyond corroborating inference.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible to available tooling (JS-rendered SPA); no bundled documentation exists in the raw archive; no academic data descriptor exists. No further realistic named source remains.

### `total_payment_value`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: True — source: Multiple convergent secondary sources (Kaggle community notebooks/tutorials paraphrasing the official Kaggle data dictionary); official Kaggle page and Olist-authored notebook could not be directly rendered/quoted by available tooling
- Evidence level: `LEVEL_4_5_CORROBORATING_NOT_PRIMARY`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED_WITH_CORROBORATION`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: Corroborating secondary evidence narrows the plausible interpretation of payment_sequential/payment_value timing but does not meet the DIRECT_PROVEN standard: no LEVEL 1-3 primary source was obtainable, and revision/backfill behavior remains completely unaddressed by any source found. Promotion standard requires ALL 8 components resolved; at least 2 remain unresolved. Multiple independent secondary sources (Kaggle community notebooks, tutorial articles) consistently paraphrase/quote wording matching Olist's known public data-dictionary language: 'payment_sequential: a customer may pay an order with more than one payment method (e.g. voucher + credit card), so a given order_id will have multiple rows for the different payment methods used in one checkout' and 'payment_value: the value of the transaction.' Independently, multiple sources consistently describe order_approved_at as 'the payment approval timestamp.' Convergent secondary corroboration (LEVEL 4-5, not independently verified against the primary Kaggle page because it could not be rendered by available tooling) suggests payment_sequential's multiplicity reflects split payment METHODS agreed together at one checkout event, not sequential future installment-charge records appended over time. This somewhat narrows -- but does not resolve -- the availability-timing concern for payment_record_count and total_payment_value: even if the full agreed value/method-split is fixed at the approval event, no evidence of any level proves the payment rows are free of later correction, refund adjustment, or backfill (Step 6 requirement 6 remains unresolved), and no directly-quotable LEVEL 1-3 primary source could be obtained to elevate this beyond corroborating inference.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible to available tooling (JS-rendered SPA); no bundled documentation exists in the raw archive; no academic data descriptor exists. No further realistic named source remains.

### `item_count`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `unique_product_count`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `unique_seller_count`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `total_item_price`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `total_freight_value`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `mean_freight_value`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `max_freight_value`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `STRONG_INFERRED`)
- New evidence found: False — source: None found beyond Execution Session 1's direct schema/code inspection
- Evidence level: `NONE_NEW`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `STRONG_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No new evidence found or obtainable. Evidence strength remains STRONG_INFERRED (order_purchase_timestamp <= order_approved_at is enforced in code), but item-level immutability between purchase and approval remains unproven because no per-item timestamp exists in the source schema. No new evidence was found or is obtainable that would resolve item/product/seller-price immutability between order_purchase_timestamp and order_approved_at; this question depends on a per-item modification timestamp that does not exist in the source schema (independently confirmed in Execution Session 1), and no external documentation search can substitute for a timestamp column that was never captured by the source system's export.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — This question is a schema fact (absence of a per-item timestamp column), already directly confirmed by source inspection in Execution Session 1; no external documentation search can substitute for a timestamp that was never captured.

### `product_category_diversity`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: False — source: None found; official Kaggle page and Olist-notebook inaccessible to available tooling
- Evidence level: `NONE_FOUND`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No evidence of any level was found addressing product-catalog snapshot/versioning semantics, despite a comprehensive primary-source search attempt. No evidence of any level was found addressing whether olist_products_dataset represents a static current-day catalog export or a point-in-time/historical snapshot. The official Kaggle dataset page and an Olist-notebook resource could not be rendered by available tooling (JS-dependent site); no bundled documentation exists in the raw archive (data/raw/olist/extracted/ contains only 9 CSVs, no README/data dictionary file); no academic data-descriptor publication exists for this dataset. This is a complete evidence absence, not a weak-but-present signal.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible; no bundled documentation exists in the raw archive; no academic data descriptor exists; repository's own provenance docs (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema semantics. No further realistic named source remains.

### `mean_product_weight_g`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: False — source: None found; official Kaggle page and Olist-notebook inaccessible to available tooling
- Evidence level: `NONE_FOUND`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No evidence of any level was found addressing product-catalog snapshot/versioning semantics, despite a comprehensive primary-source search attempt. No evidence of any level was found addressing whether olist_products_dataset represents a static current-day catalog export or a point-in-time/historical snapshot. The official Kaggle dataset page and an Olist-notebook resource could not be rendered by available tooling (JS-dependent site); no bundled documentation exists in the raw archive (data/raw/olist/extracted/ contains only 9 CSVs, no README/data dictionary file); no academic data-descriptor publication exists for this dataset. This is a complete evidence absence, not a weak-but-present signal.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible; no bundled documentation exists in the raw archive; no academic data descriptor exists; repository's own provenance docs (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema semantics. No further realistic named source remains.

### `mean_product_length_cm`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: False — source: None found; official Kaggle page and Olist-notebook inaccessible to available tooling
- Evidence level: `NONE_FOUND`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No evidence of any level was found addressing product-catalog snapshot/versioning semantics, despite a comprehensive primary-source search attempt. No evidence of any level was found addressing whether olist_products_dataset represents a static current-day catalog export or a point-in-time/historical snapshot. The official Kaggle dataset page and an Olist-notebook resource could not be rendered by available tooling (JS-dependent site); no bundled documentation exists in the raw archive (data/raw/olist/extracted/ contains only 9 CSVs, no README/data dictionary file); no academic data-descriptor publication exists for this dataset. This is a complete evidence absence, not a weak-but-present signal.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible; no bundled documentation exists in the raw archive; no academic data descriptor exists; repository's own provenance docs (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema semantics. No further realistic named source remains.

### `mean_product_height_cm`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: False — source: None found; official Kaggle page and Olist-notebook inaccessible to available tooling
- Evidence level: `NONE_FOUND`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No evidence of any level was found addressing product-catalog snapshot/versioning semantics, despite a comprehensive primary-source search attempt. No evidence of any level was found addressing whether olist_products_dataset represents a static current-day catalog export or a point-in-time/historical snapshot. The official Kaggle dataset page and an Olist-notebook resource could not be rendered by available tooling (JS-dependent site); no bundled documentation exists in the raw archive (data/raw/olist/extracted/ contains only 9 CSVs, no README/data dictionary file); no academic data-descriptor publication exists for this dataset. This is a complete evidence absence, not a weak-but-present signal.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible; no bundled documentation exists in the raw archive; no academic data descriptor exists; repository's own provenance docs (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema semantics. No further realistic named source remains.

### `mean_product_width_cm`

- Prior classification: `UNVERIFIED_ASOF` (evidence strength `WEAK_INFERRED`)
- New evidence found: False — source: None found; official Kaggle page and Olist-notebook inaccessible to available tooling
- Evidence level: `NONE_FOUND`
- Source-system availability proven: False
- Decision-time availability proven: False
- Revision/backfill risk resolved: False
- Temporal join safety resolved: True
- New evidence strength: `WEAK_INFERRED`
- **Final classification:** `UNVERIFIED_ASOF` (`primary_asof = false`)
- Reason: No evidence of any level was found addressing product-catalog snapshot/versioning semantics, despite a comprehensive primary-source search attempt. No evidence of any level was found addressing whether olist_products_dataset represents a static current-day catalog export or a point-in-time/historical snapshot. The official Kaggle dataset page and an Olist-notebook resource could not be rendered by available tooling (JS-dependent site); no bundled documentation exists in the raw archive (data/raw/olist/extracted/ contains only 9 CSVs, no README/data dictionary file); no academic data-descriptor publication exists for this dataset. This is a complete evidence absence, not a weak-but-present signal.
- `FURTHER_OLIST_EVIDENCE_SEARCH_JUSTIFIED = NO` — Official Kaggle page and an Olist-adjacent notebook were both attempted and are technically inaccessible; no bundled documentation exists in the raw archive; no academic data descriptor exists; repository's own provenance docs (docs/data_provenance.md, docs/olist_licensing_provenance_status.md) were checked and contain no additional schema semantics. No further realistic named source remains.

## External evidence sources consulted

See `EXTERNAL_EVIDENCE_LEDGER.md` in `reports/checkpoints/phase2c_asof_evidence_remediation_2026-08-09/` for the full archived ledger.

## Decisions carried forward

- `PRIMARY_FEATURE_CONTRACT = STRICT_CORE_ONLY`
- `APPROVED_EXPANDED_PRIMARY_FEATURE_COUNT = 0`
- `EXPANDED_MODEL_REEVALUATION_VALUE = LOW_OR_NONE`
- `MODEL_REEVALUATION_AUTHORIZED = FALSE`
- `TECHNICAL_DEBT_REMEDIATION_NEXT = TRUE` — the debt should be cleaned before any future model/evidence-generation pipeline reuse, but does not block unrelated NLP/data-acquisition work.