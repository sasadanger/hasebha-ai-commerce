# Existing NLP Asset Inventory

Status: repository-wide search performed this session across `src/`, `docs/`, `configs/`, `reports/`. **No NLP code, notebooks, embeddings, sentiment/text-processing scripts, or trained model artifacts exist anywhere in this repository.** No existing NLP pipeline was executed to produce this inventory.

## Assets found

| Path | Type | Purpose | Status | Reusable | Provenance known | License known | Authoritative |
|---|---|---|---|---|---|---|---|
| `data/raw/amazon_reviews_appliances/Appliances.jsonl.gz` | dataset | Raw Amazon Appliances review export | Acquired, not modeled | Yes | Yes | No | Yes |
| `data/processed/amazon_reviews_appliances/reviews_text_ready.parquet` | dataset | Cleaned/normalized Amazon Appliances reviews | Processed, not modeled | Yes | Yes | No | Yes |
| `data/profiles/amazon_reviews_appliances.json` | document | Profiling report | Reference | Yes | Yes | N/A | Yes |
| `docs/amazon_appliances_data_understanding.md` | document | Schema/quality documentation | Reference, cross-checked this session | Yes | Yes | N/A | Yes |
| `docs/amazon_reviews_voice_of_customer_readiness.md` | document | Voice-of-customer readiness assessment | Reference | Yes | Yes | N/A | Yes |
| `configs/olist_phase2c_nlp_contract.yaml` | config | Phase 2C NLP scientific contract (design-only) | Authoritative, execution deferred | Yes | Yes | N/A | Yes |

## Newly created this session (quarantine, not yet approved training data)

- `data/quarantine/nlp/labr/` — LABR dataset (downloaded, hashed, license-verified)
- `data/quarantine/nlp/astd/` — ASTD dataset (downloaded, hashed, license-verified)
- `data/quarantine/nlp/mpold/` — MPOLD dataset (downloaded, hashed, license-verified)

See `reports/generated/nlp/acquisition_manifest.json` for full file-level detail and `reports/generated/nlp/dataset_registry.json` / `configs/nlp_dataset_registry.yaml` for the complete provenance record of all 20 investigated candidates.
