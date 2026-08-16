# Phase 1A and 1B Data Pipeline

Every command selects and accesses one configured dataset domain. Olist, Instacart, and Amazon Reviews are never combined.

## Acquisition and raw inspection

```powershell
python -m src.data_pipeline.download olist --dry-run
python -m src.data_pipeline.validate_raw_data olist
python -m src.data_pipeline.profile_dataset olist
```

Valid raw-tool names are `olist`, `instacart`, and `amazon_reviews_appliances`. Existing downloads are immutable by default. Generated profiles are written below ignored `data/profiles/`.

## Phase 1B cleaning

```powershell
python -m src.data_pipeline.clean_olist --dry-run
python -m src.data_pipeline.clean_olist

python -m src.data_pipeline.clean_instacart --dry-run
python -m src.data_pipeline.clean_instacart

python -m src.data_pipeline.clean_amazon_appliances --dry-run
python -m src.data_pipeline.clean_amazon_appliances
```

Each cleaner reads only its configured raw root, writes only its configured processed root, stages output before publishing, and refuses to replace a non-empty processed directory unless `--force` is explicit. Machine-readable summaries are stored with each ignored processed dataset.

Olist and Instacart CSV tables become separate typed Parquet tables with empty fields represented as null. Amazon JSON Lines become one Parquet table that retains all records and source fields, adds a UTC datetime derived from the original epoch-millisecond value, and flags usable text without inferring sentiment.

Focused factual checks can be reproduced with:

```powershell
python -m src.data_pipeline.phase1b_observations olist
python -m src.data_pipeline.phase1b_observations instacart
python -m src.data_pipeline.phase1b_observations amazon_reviews_appliances
```

These reports remain under ignored `data/profiles/`. Phase 1B produces no feature matrix, model, recommendation, sentiment label, or prediction.
