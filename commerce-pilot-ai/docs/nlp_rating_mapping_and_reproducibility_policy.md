# Amazon/LABR Rating-to-Sentiment Mapping Policy & Reproducibility Policy

`PHASE2C_AUTHORIZATION_SCOPE = NLP_EXPERIMENT_DEFINITION_ONLY`. Design only.

## Step 27 — Star-rating-to-sentiment mapping policy

Amazon Appliances and LABR both carry native 1–5 star ratings, not a native sentiment label. Whether a future experiment treats rating as `REVIEW_RATING` (its native task, per `configs/nlp_task_dataset_matrix.yaml`) or additionally derives a `SENTIMENT` label from it is **not decided by default** in this gate.

If a future experiment chooses to derive sentiment from rating, the mapping must be stated explicitly per-experiment, e.g.:

```
1-2 stars -> negative
3 stars   -> neutral
4-5 stars -> positive
```

This mapping is **documented here as a candidate, not adopted automatically**. Before any future experiment uses it:

1. The experiment manifest entry must name the exact mapping used (not just "rating-derived sentiment").
2. The ambiguity of 3-star reviews (genuinely neutral vs. mildly positive/negative) must be acknowledged in that experiment's `what_it_does_not_prove` field.
3. `configs/nlp_task_dataset_matrix.yaml`'s `mapping_type: PARTIAL` designation for this mapping must not be silently upgraded to `EXACT`.

Both Experiment A and Experiment C currently use `REVIEW_RATING` as their native task, not a derived sentiment label, precisely to avoid adopting this mapping without an explicit future decision.

## Step 28/29 — Duplicate re-audit requirements (cross-referenced, canonical statement lives in the duplicate contract)

`EGYPTIAN_TWEETS_DUPLICATE_REAUDIT_REQUIRED = YES` and `ARSAS_DUPLICATE_REAUDIT_REQUIRED = YES` are declared in `configs/nlp_duplicate_control_contract.yaml#mandatory_reaudits_before_any_future_split`. Old duplicate counts (379/13 for Egyptian Tweets 40K; 99/31 for ArSAS, per the remediation session's `quarantine_quality_audit_v2.json`) are explicitly **not** trusted as final split input — `configs/nlp_split_policy.yaml` marks `EXPERIMENT_B1` and `EXPERIMENT_D1`/`EXPERIMENT_D2` as `BLOCKED_PENDING_DUPLICATE_REAUDIT` in the experiment manifest until a recomputation using the canonical `NORMALIZED_EXACT_KEY` (defined in the duplicate contract) is performed.

## Step 24 — Reproducibility policy

Every future experiment run must record, alongside its results:

| Field | Requirement |
|---|---|
| Data split seed | `20260809` (fixed, per `configs/nlp_split_policy.yaml`) unless a documented reason requires a different seed, in which case the new seed is recorded per-run |
| Model seed | Recorded per-run; must differ from the split seed to avoid conflating data and model randomness |
| Library versions | Full `pip freeze` (or equivalent) snapshot recorded per-run |
| Hardware | CPU/GPU model and count recorded per-run |
| Tokenizer version | Recorded per-run, including any transformer tokenizer's exact model-revision hash |
| Model revision | Exact commit/checkpoint hash for any pretrained model used |
| Dataset hash | SHA-256 of the exact input file(s) used, per `reports/generated/nlp/acquisition_manifest_v2.json`-style hashing |
| Config hash | SHA-256 of the exact `configs/nlp_experiment_manifest.yaml` entry (and any referenced contract file) used for that run |

No future training run may be treated as reproducible evidence without all eight fields recorded.
