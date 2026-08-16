# Dataset Card: Jumia Egypt Reviews Dataset (JERD)

`dataset_id = jumia_egypt_reviews_jerd` | Tier: `QUARANTINE` | Approval: `QUARANTINE_PROVENANCE`

## Purpose
Domain: e-commerce product reviews -- direct domain and country match. Commerce relevance: HIGH.

## Source and provenance
- Publisher/authors: Kaggle user aliallam13 (NOT Jumia itself; unofficial third-party upload) / unspecified (n/a)
- Primary source: https://www.kaggle.com/datasets/aliallam13/jumia-egypt-reviews
- Provenance status: `VERIFIED_LEVEL_5 (individual Kaggle uploader; collection/scraping methodology not documented; no confirmed link to Jumia itself)`

## Language and platform
- Languages: multilingual (Arabic and/or English -- not independently confirmed)
- Dialects: unspecified
- Egypt-specific: True
- Platforms: Jumia (real Egyptian/African e-commerce marketplace)

## Domain, labels, size
- Tasks: sentiment/rating analysis
- Labels: unspecified
- Sample count reported: unknown exact row count; file reported as 229 KB single CSV
- Sample count independently verified: None

## License and commercial-use status
- License reported: UNKNOWN (Kaggle dataset page could not be rendered by available WebFetch tooling)
- License verified: False
- Commercial-use status: `UNKNOWN`
- Redistribution status: UNKNOWN

## Quality findings
- Schema status: NOT_INSPECTED
- Duplicate status: NOT_MEASURED
- Language audit: NOT_PERFORMED

## PII findings
- PII risk: UNKNOWN

## Known limitations
Small and topically ideal (real Egyptian e-commerce marketplace), but provenance is an unverified individual Kaggle upload with no documented collection methodology, and Kaggle's page could not be rendered by available tooling to check the stated license (same JavaScript-rendering barrier encountered throughout this and the prior Olist AS-OF gate).

## Recommended role
unspecified

## Approval status
`QUARANTINE_PROVENANCE` (Tier `QUARANTINE`)

## Review notes
Genuinely promising discovery given the domain/country match and small safe size (229 KB), but provenance quality is the lowest tier (LEVEL_5) found among promising candidates this session. Worth a targeted Kaggle-API-based recovery attempt (rather than browser-style WebFetch) in a future session.

## Citation / evidence sources
- Kaggle search listing (page content itself inaccessible)
