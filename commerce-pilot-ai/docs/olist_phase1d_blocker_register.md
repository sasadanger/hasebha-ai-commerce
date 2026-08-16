# Olist Phase 1D Blocker Register

Reassessment date: 2026-08-03. Evidence statuses use the Phase 1D controlled vocabulary.

| ID | Original blocker | Why it matters | Evidence | Status | Technical / legal / production consequence | Resolution and remaining action |
|---|---|---|---|---|---|---|
| OLI-01 | Stored estimate not proven immutable at approval | Could overstate a customer-promise-at-approval claim | Processed schema; owner Kaggle dataset page; Phase 1C audit | UNVERIFIED | Does not invalidate an honestly named retrospective stored-estimate benchmark; blocks promised-lead-time inputs and approval-time-promise claims | Target renamed to retrospective stored-estimate exceedance. Obtain owner/data-dictionary evidence before stronger claim. |
| OLI-02 | Future actual delivery used in target | Could be confused with feature leakage | Target code/contract/tests | VERIFIED_FROM_REPOSITORY | None when label-only; severe leakage if input | Target generated after eligibility; actual delivery is programmatically target-only. Resolved. |
| OLI-03 | Final-export snapshots not proven available at approval | Item/payment/catalog/location features could contain later state | Parquet schemas and absence of snapshot metadata | UNVERIFIED | Blocks these features from primary benchmark | Isolated to expanded sensitivity contract; verify source-system snapshot semantics before promotion. |
| OLI-04 | No meaningful safe feature set | Would make an offline benchmark trivial or invalid | Purchase and approval timestamps; typed schema; anomaly exclusions | SUPPORTED_INFERENCE | Nine strict calendar/timing features enable a narrow benchmark | Resolved for local offline research; do not imply operational completeness. |
| OLI-05 | Missing recall target and costs | Prevents an operationally optimal threshold | Evaluation docs/config | VERIFIED_FROM_REPOSITORY | Does not block threshold-independent training/evaluation; blocks deployment threshold | Report PR-AUC and threshold tables; no optimal production threshold claim. |
| OLI-06 | License/version uncertainty | May restrict use/publication/commercialization | Owner dataset page; provenance; no local license/version manifest | VERIFIED_FROM_PRIMARY_SOURCE for displayed license; UNVERIFIED for exact version attachment | Legal NO-GO remains | Legal review; record exact version and license artifact. |
| OLI-07 | Brazilian-to-Egypt transfer | Could misrepresent production validity | Dataset description and observed geography/time | VERIFIED_FROM_PRIMARY_SOURCE / VERIFIED_FROM_REPOSITORY | Does not create leakage; blocks Egyptian performance claims and deployment | Require Egyptian live-store data and validation. |

