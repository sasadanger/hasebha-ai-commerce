# Olist Strict Offline Experiment Contract

This is the primary future scientific benchmark. It uses eligibility policy `olist-eligibility-v1`, anomaly policy `olist-timestamp-anomaly-v1`, feature contract `olist-feature-contract-v1`, and the frozen approval-time split.

Strict features only:

- purchase year, month, day of week, and hour;
- approval year, month, day of week, and hour;
- purchase-to-approval duration in seconds.

They derive only from `order_purchase_timestamp` and `order_approved_at`. Raw identifiers are audit/join keys, not inputs. Stored estimate and actual delivery construct the retrospective label only. No item, payment, product, seller, customer-location, geolocation, historical-outcome, status, shipping, or review feature enters Contract A.

All learned preprocessing fits on training only. The final test remains untouched until the entire pipeline, model choice process, calibration plan, and validation-selected threshold rule are frozen. PR-AUC is primary. Missing costs do not block ranking comparison; they prohibit an “optimal production threshold” claim.

This contract permits only a local offline benchmark after legal/use authorization is separately addressed. It creates no Egyptian-production claim.

