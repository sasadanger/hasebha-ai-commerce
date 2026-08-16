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
