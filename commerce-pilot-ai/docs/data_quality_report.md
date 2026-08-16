# Phase 1B Data Quality Report

## Olist

Nine inputs produced nine Parquet outputs with identical per-table row counts: 99,441; 1,000,163; 112,650; 103,886; 99,224; 99,441; 32,951; 3,095; and 71 in documented file order. Zero rows were dropped. Candidate order, customer, item, product, seller, and translation keys had no duplicates. Missingness is concentrated in review text, three order-event timestamps, and product descriptive/physical fields. Timestamp anomalies were retained. Parsing completed without reported failure.

## Instacart

Six inputs produced six Parquet outputs with unchanged counts: 134; 21; 32,434,489; 1,384,617; 3,421,083; and 49,688. Zero rows were dropped. All tested candidate keys had zero duplicates. Only `days_since_prior_order` had missing values (206,209). Hour values all parsed to integers in 0–23; taxonomy orphan checks returned zero. Archive provenance and license limitations remain.

## Amazon Reviews 2023 — Appliances

The input and output both contain 2,128,605 records; zero were dropped and no malformed record was skipped. All source fields had zero parsed nulls. There are 2,065 empty/whitespace texts and 22,657 duplicate rows by the documented candidate identity; both are retained. Ratings were limited to observed values 1–5 and helpful votes were nonnegative. The original timestamp is retained and a UTC datetime is added.

No combined metrics or cross-dataset calculations were performed. Machine-readable details are in each ignored `data/processed/<dataset>/cleaning_summary.json` and `data/profiles/phase1b_<dataset>.json`.

