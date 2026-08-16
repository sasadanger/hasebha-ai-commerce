# Dataset Card: Amazon Reviews 2023 -- Appliances

`dataset_id = amazon_appliances` | Tier: `TIER_A_CORE` | Approval: `APPROVED_FOR_BENCHMARK_ONLY`

## Purpose
Domain: e-commerce product reviews (Appliances category). Commerce relevance: HIGH.

## Source and provenance
- Publisher/authors: McAuley Lab, UC San Diego / McAuley Lab (Hou et al.) (2023)
- Primary source: https://amazon-reviews-2023.github.io/main.html
- Provenance status: `VERIFIED_LEVEL_1`

## Language and platform
- Languages: English
- Dialects: N/A
- Egypt-specific: False
- Platforms: Amazon marketplace

## Domain, labels, size
- Tasks: sentiment (unlabeled, ratings only), review-text methodology baseline
- Physical label: `rating` (1-5); canonical NLP label after the explicit adapter: `overall` (not sentiment-labeled)
- Sample count reported: 2128605
- Sample count independently verified: 2128605

## License and commercial-use status
- License reported: Not stated on verified project page
- License verified: False
- Commercial-use status: `UNKNOWN`
- Redistribution status: UNKNOWN

## Quality findings
- Schema status: VERIFIED_THIS_SESSION
- Duplicate status: 287,718 exact-duplicate text rows (13.5%) out of 2,128,605 (independently measured this session); 22,657 duplicate rows on the physical (user_id,parent_asin,timestamp,rating,text) composite key (previously measured, docs/amazon_appliances_data_understanding.md)
- Language audit: PERFORMED_THIS_SESSION: 5,000-row sample, 0.00% contains Arabic-script characters, 99.72% contains Latin characters -- confirms English-only, no Arabic content.

## PII findings
- PII risk: MEDIUM (pseudonymous user_id, free text may contain incidental personal narrative)

## Known limitations
None beyond what is stated above.

## Recommended role
ENGLISH_ECOMMERCE_BENCHMARK, MULTIPLATFORM_ROBUSTNESS_BENCHMARK

## Approval status
`APPROVED_FOR_BENCHMARK_ONLY` (Tier `TIER_A_CORE`)

## Review notes
Pre-existing project asset (Phase 1A), not newly acquired this session. Confirmed independently: hash matches provenance record, schema unchanged, English-only. Does NOT prove Egyptian-market, Arabic, or code-switch capability -- explicitly preserved as PRESERVED_NOT_CURRENTLY_EXECUTED per governing prompt.

## Citation / evidence sources
- docs/amazon_appliances_data_understanding.md
- docs/amazon_reviews_voice_of_customer_readiness.md
- docs/data_provenance.md
- direct local schema/hash re-verification this session
