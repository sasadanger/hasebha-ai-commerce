# Data Acquisition Plan

## Purpose and boundaries

Phase 1A acquires and inspects Olist, Instacart, and Amazon Reviews 2023 Appliances as three independent data domains. No cross-dataset join, entity matching, shared feature construction, or comparative inference is permitted. Their only planned future connection is the Decision Action Card API contract.

## Source-verification checklist

Before any download:

- Confirm the dataset name and intended capability against the approved register.
- Prefer the owner or research publisher's page and its documented hosting provider.
- Record the exact source page, direct download URL, owner or source organization, and access date.
- Record a stated version and checksum only when the source provides them.
- Review the license, competition rules, usage terms, attribution, and access restrictions.
- Confirm the download does not require bypassing authentication, interactive terms, payment, CAPTCHA, or another access control.
- Mark information that cannot be established as `Not verified`; never infer it.

## Download policy

- A command must explicitly name exactly one approved dataset.
- URLs and raw destinations come from YAML configuration.
- Downloads go only to the selected dataset's configured raw directory.
- Official or clearly documented public sources are required; altered or undocumented mirrors are prohibited.
- Dry runs must make no network request and create no files.
- Existing raw files must not be overwritten unless `--force` is explicitly supplied.
- Failed downloads must record the attempted source, action, failure, and whether user action is required.
- Credentials, tokens, and secrets must never be stored in configuration or Git.

## Immutable raw-data policy

Raw downloads and extracted source files are evidence. After successful acquisition they are read-only inputs in normal workflows. Corrections and transformations belong under capability-specific processed paths, never in raw files. A replacement requires an explicit force operation, a documented reason, refreshed checksums, and updated provenance.

Raw data, downloaded archives, generated profiles, and logs remain outside Git through `.gitignore` rules.

## Recording policy

For every attempt, provenance records must state the access date, source owner, exact URLs, stated version or `Not verified`, terms or `Not verified`, acquisition result, local destination, and blocker. After a successful download, calculate and record a local SHA-256 checksum. A publisher checksum is recorded separately only if one is actually supplied.

## Privacy and licensing

Public accessibility does not imply unrestricted use. Applicable license and usage terms must be reviewed before downstream work. Access should be limited to authorized project work, retained data minimized, and raw text treated as potentially sensitive even when a source describes data as anonymized. Profiles should report structural facts without reproducing review text or identifiers.

## Dataset-specific acquisition goals

- **Olist:** acquire the Olist-published Brazilian E-Commerce Public Dataset from its documented Kaggle record for fulfillment and delivery research.
- **Instacart:** acquire the Instacart Market Basket Analysis competition data only through its documented Kaggle competition source and accepted access terms.
- **Amazon Reviews 2023 Appliances:** acquire only the Appliances review category from the McAuley Lab project; do not acquire other categories or all-category bundles.

## Stop condition

No downstream feature engineering or modeling begins until provenance is complete, the applicable data-use terms have been reviewed, and raw validation has passed for the relevant independent dataset.

