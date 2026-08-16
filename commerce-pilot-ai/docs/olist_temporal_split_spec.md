# Olist Temporal Split Specification

Split variable: `order_approved_at`. Boundaries are frozen in `configs/olist_modeling.example.yaml`; intervals are start-inclusive and end-exclusive.

| Split | Calendar interval | Observed approval range | Orders | Positive | Negative | Prevalence |
|---|---|---|---:|---:|---:|---:|
| Train | 2016-09-01 to 2018-01-01 | 2016-09-15 12:16:38 to 2017-12-31 23:32:40 | 43,516 | 2,875 | 40,641 | 6.606765% |
| Validation | 2018-01-01 to 2018-05-01 | 2018-01-01 02:30:25 to 2018-04-30 21:29:47 | 26,822 | 3,351 | 23,471 | 12.493476% |
| Test | 2018-05-01 to 2018-09-01 | 2018-05-01 01:56:05 to 2018-08-29 15:10:26 | 24,744 | 1,566 | 23,178 | 6.328807% |

All 95,082 eligible orders fall in exactly one window. The boundaries are contiguous, chronological calendar boundaries chosen before model scores. Monthly prevalence varies materially, including 21.61% in March 2018 and 1.37% in June 2018; this is a reason to retain temporal evaluation, not alter dates.

Against training, validation contains 25,902 unseen customers, 547 unseen sellers, 7,235 unseen products, and zero unseen non-null categories. Test contains 24,138 unseen customers, 1,095 unseen sellers, 9,519 unseen products, and one unseen non-null category. Unknown-safe transformations are mandatory.

The final eligible approval is August 29, 2018. No eligible rows exist outside the frozen windows; September/October purchase records in the full table do not qualify for this delivered, approval-based cohort. The test set remains untouched during preprocessing design, feature selection, calibration, threshold selection, and tuning.

