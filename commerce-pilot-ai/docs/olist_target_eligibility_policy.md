# Olist Target and Eligibility Policy

Policy version: `olist-eligibility-v1`. Prediction point: confirmed payment approval. Unit: one order.

## Verified target

Both `order_delivered_customer_date` and `order_estimated_delivery_date` exist as parsed timestamps in `olist_orders_dataset.parquet`. The proposed retrospective binary label is:

```text
late_delivery = order_delivered_customer_date > order_estimated_delivery_date
```

Actual delivery is used only to construct the label and is forbidden as a feature. The formula means “delivered after the recorded estimate,” not cancellation, non-delivery, customer dissatisfaction, or generalized fulfillment risk. The repository does not prove whether the recorded estimate is the immutable promise visible at payment approval; this blocks training until verified.

## Sequential exclusion waterfall

Every source order is assigned to the first applicable rule. Percentages use 99,441 original orders.

| Rule | Treatment | Excluded | % original | Remaining | Positive | Negative | Prevalence |
|---|---|---:|---:|---:|---:|---:|---:|
| Duplicate `order_id` | Exclude all affected IDs | 0 | 0.000000% | 99,441 | 7,827 | 88,649 | 8.112899% of labeled |
| Status is not `delivered` | Exclude primary; separate future non-delivery task | 2,963 | 2.979656% | 96,478 | 7,826 | 88,644 | 8.112367% of labeled |
| Missing approval time | Exclude: prediction point absent | 14 | 0.014079% | 96,464 | 7,826 | 88,630 | 8.113544% of labeled |
| Missing actual delivery | Exclude: label unavailable | 8 | 0.008045% | 96,456 | 7,826 | 88,630 | 8.113544% |
| Missing estimated delivery | Exclude: label unavailable | 0 | 0.000000% | 96,456 | 7,826 | 88,630 | 8.113544% |
| Approval before purchase | Exclude impossible sequence | 0 | 0.000000% | 96,456 | 7,826 | 88,630 | 8.113544% |
| Delivery before purchase | Exclude invalid outcome time | 0 | 0.000000% | 96,456 | 7,826 | 88,630 | 8.113544% |
| Estimate before approval | Exclude primary; sensitivity cohort | 6 | 0.006034% | 96,450 | 7,823 | 88,627 | 8.110938% |
| Carrier before approval | Exclude primary; sensitivity cohort | 1,345 | 1.352561% | 95,105 | 7,792 | 87,313 | 8.193050% |
| Delivery before carrier | Exclude primary; sensitivity cohort | 23 | 0.023129% | 95,082 | 7,792 | 87,290 | 8.195032% |

Before the missing-delivery rule, `unlabeled` counts are 2,965 initially and eight after restricting to delivered orders; positives plus negatives therefore do not equal remaining orders at those stages.

Canceled and unavailable orders are never labeled “on time.” Multiple items, sellers, or payments do not exclude an order; they require deterministic aggregation. One eligible order has no payment row and is retained with explicit aggregate-missing indicators. Orders with missing carrier time are retained because carrier time is neither an input nor required for the target.

Sensitivity analyses may separately reintroduce estimate-before-approval, carrier-before-approval, or delivery-before-carrier cohorts and must report them against the unchanged primary cohort. Missing labels and non-delivered statuses remain outside this binary task. No date is repaired or invented.

