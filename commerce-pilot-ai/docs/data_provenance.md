# Data Provenance

Last updated: 2026-08-03. Records below distinguish source-page verification from actual acquisition results. `Not verified` means the information was not established from the verified source or acquisition attempt.

## Olist — fulfillment and delivery intelligence

- **Dataset:** Brazilian E-Commerce Public Dataset by Olist
- **Source organization/owner:** Olist
- **Source page:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Direct URL attempted:** https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce
- **Access date:** 2026-08-03
- **Version:** Not verified
- **License/usage terms:** CC BY-NC-SA 4.0, as stated on the Kaggle dataset page
- **Publisher checksum:** Not verified
- **Acquisition status:** Successful
- **Action attempted:** `python -m src.data_pipeline.download olist`
- **Saved archive:** `data/raw/olist/brazilian-ecommerce.zip` (44,717,580 bytes observed locally)
- **Extracted destination:** `data/raw/olist/extracted/` (nine CSV files observed)
- **Local SHA-256:** `967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784`
- **Validation status:** Passed; the ZIP integrity check passed and all ten raw/archive files were readable and non-empty
- **Profiling status:** Completed; factual JSON report saved at ignored path `data/profiles/olist.json`
- **Blocker and user action:** None for acquisition. License suitability for the project's eventual use must still be reviewed because the stated license is non-commercial and share-alike.

## Olist Marketing Funnel — research enrichment (added 2026-08-22)

- **Dataset:** Marketing Funnel by Olist (alternate name: "8k leads, closed deals and connection
  to 100k orders")
- **Source organization/owner:** Olist
- **Source page:** https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist
- **Direct URL used:** https://www.kaggle.com/api/v1/datasets/download/olistbr/marketing-funnel-olist
- **Access date:** 2026-08-22
- **Version:** datasetVersionNumber=2 (per the page's own JSON-LD metadata, `contentSize` 284,562
  bytes matches the downloaded archive exactly)
- **License/usage terms:** CC BY-NC-SA 4.0 (per the page's JSON-LD `license` field), same terms
  as the main Olist e-commerce dataset — same NO-GO-until-reviewed licensing status applies (see
  `docs/olist_licensing_provenance_status.md`); this addition is RESEARCH-ONLY, not a change to
  that licensing verdict.
- **Publisher checksum:** Not published by Kaggle; local SHA-256 recorded below instead.
- **Acquisition status:** Successful, anonymous (no Kaggle credentials were configured or used
  in this environment; the dataset's public download endpoint served the archive directly over
  HTTPS without authentication — confirmed by inspecting the returned bytes as a genuine ZIP
  archive, not a login page).
- **Saved archive:** `data/raw/olist_funnel/marketing-funnel-olist.zip` (284,562 bytes)
- **Local SHA-256 (archive):** `d476efe8d16ab267ce10535859ed3dd2d6e186bbb50c9d405a1e7bc05c200171`
- **Extracted destination:** `data/raw/olist_funnel/` (2 CSV files)
- **Local SHA-256 (olist_closed_deals_dataset.csv):** `1433f402d8fff00ca167579de641d9525f6e846368958c166f699ef1b0c31f3d`
- **Local SHA-256 (olist_marketing_qualified_leads_dataset.csv):** `5537753b7549ce4068e52e2402c29f2773c5375fc0575520ae8f3f301f666573`
- **Validation status:** Passed — ZIP integrity confirmed via successful extraction; both CSVs
  readable, correct column sets, no duplicate primary keys (`mql_id` in both files, `seller_id`
  in closed_deals).
- **Row counts (actual, verified by direct read, not the mission prompt's approximate figures):**
  `olist_marketing_qualified_leads_dataset.csv` = 8,000 rows, 4 columns.
  `olist_closed_deals_dataset.csv` = 842 rows, 14 columns.
- **Date ranges (actual, verified):** MQL `first_contact_date` 2017-06-14 to 2018-05-31.
  Closed-deals `won_date` 2017-12-05 to 2018-11-14.
- **PII risk:** Low — no customer PII; contains internal Olist identifiers (`sdr_id`, `sr_id`)
  and seller-level declared business data (`declared_monthly_revenue`, `business_segment`),
  no personal names/emails/addresses observed in a column-level review.
- **Data quality flags:** `declared_monthly_revenue` is unreliable — 797 of 842 rows (94.7%) are
  exactly 0, so it is treated as effectively non-informative (excluded as a numeric feature,
  see Phase 2 audit). `has_company`/`has_gtin`/`average_stock`/`declared_product_catalog_size`
  are 92%+ null and not used. `lead_behaviour_profile` is 21% null.
- **Join coverage against the Olist V3 seller-SLA canonical cohort (96,380 rows, 2,955 unique
  sellers):** only 377 of 2,955 canonical sellers (12.8%) appear in `closed_deals`, covering
  4,384 of 96,380 canonical rows (4.5%). This is a structural ceiling on any predictive value
  this dataset can contribute (see `reports/generated/olist_funnel/FUNNEL_DATA_QUALITY_AUDIT.json`
  for full detail) — most of the canonical cohort's date range (2016-09 to 2018-08) predates or
  falls outside the funnel dataset's own coverage window (mid-2017 to end-2018).
- **Blocker and user action:** None for acquisition. Same licensing caveat as the base Olist
  dataset applies before any non-research use.

## Instacart — personalization and recommendation

- **Dataset:** Instacart Market Basket Analysis
- **Source organization/owner:** Instacart; competition data hosted by Kaggle
- **Source page:** https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
- **Direct URL attempted:** https://www.kaggle.com/api/v1/competitions/data/download-all/instacart-market-basket-analysis
- **Access date:** 2026-08-03
- **Version:** Not verified
- **License/usage terms:** Not verified; Kaggle competition rules and access terms apply
- **Publisher checksum:** Not verified
- **Acquisition status:** Available locally through manual placement. The earlier automatic attempt against the documented Kaggle endpoint returned HTTP 401 Unauthorized; it was not retried.
- **Action history:** `python -m src.data_pipeline.download instacart` returned HTTP 401; the user subsequently placed `archive_2.zip` in the configured raw directory; the archive was extracted locally without overwrite.
- **Archive origin:** Not independently verified from the file itself
- **Saved archive:** `data/raw/instacart/archive_2.zip` (207,073,669 bytes observed locally)
- **Extracted destination:** `data/raw/instacart/extracted/` (six CSV files observed)
- **Local SHA-256:** `c347cd7fa301c068ed391c7c77620f0e26e18f65d04cb907b6389c670db03038`
- **Validation status:** Passed; the ZIP integrity check passed and all seven raw/archive files were readable and non-empty
- **Profiling status:** Completed; factual JSON report saved at ignored path `data/profiles/instacart.json`
- **Observed profile summary:** six CSV files; observed row counts excluding headers are 134 (`aisles.csv`), 21 (`departments.csv`), 32,434,489 (`order_products__prior.csv`), 1,384,617 (`order_products__train.csv`), 3,421,083 (`orders.csv`), and 49,688 (`products.csv`)
- **Blocker and user action:** No remaining blocker for local validation and profiling. License/usage terms, version, publisher checksum, and the manually supplied archive's origin remain Not verified and must be resolved before downstream use.

## Amazon Reviews 2023 — Appliances only — voice of customer

- **Dataset:** Amazon Reviews 2023, Appliances review category only
- **Source organization/owner:** McAuley Lab, University of California San Diego
- **Source page:** https://amazon-reviews-2023.github.io/main.html
- **Direct URL attempted:** https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz
- **Access date:** 2026-08-03
- **Version:** Amazon Reviews 2023
- **License/usage terms:** Not verified; no dataset license was identified on the verified project page
- **Publisher checksum:** Not verified
- **Acquisition status:** Successful. An initial inferred host URL (`https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz`) returned HTTP 404; the exact hyperlink on the official project page was then verified and used successfully.
- **Action attempted:** `python -m src.data_pipeline.download amazon_reviews_appliances`
- **Saved file:** `data/raw/amazon_reviews_appliances/Appliances.jsonl.gz` (270,209,794 bytes observed locally)
- **Local SHA-256:** `150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2`
- **Validation status:** Passed; gzip decompression succeeded and the first record parsed as JSON
- **Profiling status:** Completed; 2,128,605 JSON Lines records were observed and a factual report was saved at ignored path `data/profiles/amazon_reviews_appliances.json`
- **Blocker and user action:** No acquisition blocker. License or usage terms remain Not verified and require review before downstream use.

## Independence statement

These provenance records and their raw directories are independent. No record implies common customers, products, orders, or identities across datasets. Future integration is limited to the shared Decision Action Card API contract.
