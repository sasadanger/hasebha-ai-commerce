# Olist Evidence Register

## Repository evidence

| Claim | Evidence | Status |
|---|---|---|
| Nine processed Olist schemas and hashes | `data/processed/olist/*.parquet`; generated audits | VERIFIED_FROM_REPOSITORY |
| Primary cohort is 95,082 with 7,792 positives and 87,290 negatives | `reports/generated/olist/readiness_audit.json` and Phase 1D recomputation | VERIFIED_FROM_REPOSITORY |
| Target components are typed timestamps and absent from strict inputs | Processed orders schema; feature contract; tests | VERIFIED_FROM_REPOSITORY |
| Equality is negative; missing components are excluded before labeling | Audit implementation and fixture tests | VERIFIED_FROM_REPOSITORY |
| Strict calendar/timing features use events no later than approval | Purchase and approval field semantics/names and chronological policy | SUPPORTED_INFERENCE |
| Stored estimate was immutable and customer-visible at approval | No qualifying evidence found | UNVERIFIED |
| Item/payment/catalog/location values are approval-time snapshots | No version or snapshot fields found | UNVERIFIED |
| Retrospective stored-estimate exceedance is a coherent supervised outcome | Label logic, eligible delivered cohort, non-overlapping temporal split | SUPPORTED_INFERENCE |
| Business costs/minimum recall | Not present | UNVERIFIED |

## External sources inspected

Access date: 2026-08-03.

1. **Brazilian E-Commerce Public Dataset by Olist** — publisher: Olist, hosted by Kaggle — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce — primary dataset record. Supports dataset identity, anonymized historical Brazilian marketplace scope, 2016–2018 coverage, multiple-item/seller warning, delivery-performance research purpose, and displayed CC BY-NC-SA 4.0 license. It calls the field an estimated delivery date but does not establish approval-time immutability.
2. **Attribution-NonCommercial-ShareAlike 4.0 International deed** — publisher: Creative Commons — https://creativecommons.org/licenses/by-nc-sa/4.0/ — primary license authority. Supports attribution, noncommercial, and share-alike conditions and cautions that other permissions may be necessary.

No blog, tutorial, notebook, forum, or altered mirror was used as proof. The exact downloaded dataset version remains UNVERIFIED.

