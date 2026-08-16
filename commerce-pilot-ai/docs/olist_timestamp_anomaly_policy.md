# Olist Timestamp-Anomaly Policy

Policy version: `olist-timestamp-anomaly-v1`. Counts below are independent checks across all 99,441 order rows; overlapping records can appear in multiple checks.

| Anomaly | Count | % orders | Availability/target impact | Primary treatment | Sensitivity treatment |
|---|---:|---:|---|---|---|
| Approval missing | 160 | 0.160899% | Prediction time absent; target may still exist | Exclude when reached (14 delivered orders) | None for payment-approval task |
| Carrier before approval | 1,359 | 1.366637% | Prediction would occur after handoff; violates intended decision timing | Exclude when reached (1,345) | Evaluate as separately flagged cohort |
| Delivery before carrier | 23 | 0.023129% | Outcome sequence is implausible; target timestamp credibility affected | Exclude all 23 when reached | Separately flagged cohort only |
| Delivery before purchase | 0 | 0% | Would invalidate outcome time | Exclude if encountered | None |
| Estimate before approval | 12 | 0.012068% | Recorded promise precedes prediction time and may imply invalid promised lead time | Exclude when reached (6) | Separately flagged cohort |
| Approval before purchase | 0 | 0% | Invalid prediction chronology | Exclude if encountered | None |
| Carrier before purchase | 166 | 0.166930% | Additional lifecycle violation; overlaps other rules | Captured by conservative sequence rules | Report separately |
| Estimate before purchase | 0 | 0% | Would undermine recorded promise | Exclude if encountered | None |
| Invalid/unparsable timestamp | 0 | 0% | Phase 1B Parquet schema contains typed timestamps | Fail audit on future schema/parse drift | None |

The policy never swaps dates, imputes event times, clips durations, or treats anomalies as normal. Counts are recomputed by `readiness_audit.py`.

