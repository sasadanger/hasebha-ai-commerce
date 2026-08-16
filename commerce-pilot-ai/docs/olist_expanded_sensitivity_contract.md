# Olist Expanded Sensitivity Contract

Contract B is secondary and must never be mixed with the strict benchmark. It adds only `CONDITIONALLY_ALLOWED` features from `olist_feature_contract_v1.yaml`, including payment, item, commercial, product, seller/location, distance, and past-only historical aggregates.

Every run must state that final-export fields may not represent approval-time snapshots. Historical statistics must use outcomes strictly before each approval and be fit independently inside temporal training folds. Promised lead time remains unresolved and is excluded. Status, shipping-limit/handoff, delivery, reviews, target derivatives, and validation/test outcome aggregates remain forbidden.

Contract B must reuse the same cohort, splits, metrics, candidate models, and test protection as Contract A. Its report must be labeled “expanded snapshot-assumption sensitivity experiment,” compare against Contract A, and avoid attributing any gain to production-available information without new evidence.

