# Dataset Card: HARD: Hotel Arabic-Reviews Dataset

`dataset_id = hard_hotel_reviews` | Tier: `QUARANTINE` | Approval: `QUARANTINE_LICENSE`

## Purpose
Domain: hotel reviews -- general review-sentiment methodology, not e-commerce/delivery. Commerce relevance: MEDIUM (review-sentiment methodology transfers; domain is hospitality, not e-commerce).

## Source and provenance
- Publisher/authors: Springer (Elnagar, Khalifa, Einea 2018) / Ashraf Elnagar, Yasmin S. Khalifa, Anas Einea (2018)
- Primary source: https://github.com/elnagara/HARD-Arabic-Dataset
- Provenance status: `VERIFIED_LEVEL_2 (author's own GitHub + peer-reviewed book chapter)`

## Language and platform
- Languages: Arabic
- Dialects: MSA and dialectal Arabic, mixed
- Egypt-specific: False
- Platforms: Booking.com

## Domain, labels, size
- Tasks: sentiment classification
- Labels: rating (1-5)
- Sample count reported: 93,700 (balanced) -- CONTRADICTORY figures found for the full/unbalanced set (373,750 vs 490,587 reported across different secondary sources)
- Sample count independently verified: None

## License and commercial-use status
- License reported: NO_LICENSE_FOUND (no LICENSE file or statement located in the repository this session)
- License verified: False
- Commercial-use status: `PROHIBITED_PENDING_REVIEW (no repo license found; underlying content originates from Booking.com, a commercial travel platform whose own ToS govern review-content rights and were not reviewed this session)`
- Redistribution status: UNVERIFIED

## Quality findings
- Schema status: NOT_INSPECTED
- Duplicate status: NOT_MEASURED
- Language audit: NOT_PERFORMED

## PII findings
- PII risk: UNKNOWN

## Known limitations
No repository license found; source-platform (Booking.com) terms not reviewed; conflicting sample-count figures across sources not reconciled. Held in quarantine pending a dedicated license/rights review, consistent with governing instructions for HARD specifically.

## Recommended role
unspecified

## Approval status
`QUARANTINE_LICENSE` (Tier `QUARANTINE`)

## Review notes
Exactly the code-license-vs-source-platform-terms distinction the governing prompt warned about: no code license was even found, and Booking.com's own terms remain a separate, unreviewed risk.

## Citation / evidence sources
- https://github.com/elnagara/HARD-Arabic-Dataset (fetched directly)
- https://link.springer.com/chapter/10.1007/978-3-319-67056-0_3
