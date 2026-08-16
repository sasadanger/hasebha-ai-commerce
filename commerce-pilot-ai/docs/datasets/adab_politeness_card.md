# Dataset Card: ADAB: Arabic Dataset for Automated Politeness Benchmarking

`dataset_id = adab_politeness` | Tier: `TIER_A_CORE` | Approval: `ACCESS_PENDING`

## Purpose
Domain: politeness classification across social media, e-commerce, and customer-service text -- directly relevant domain match. Commerce relevance: HIGH (explicit e-commerce and customer-service platform coverage).

## Source and provenance
- Publisher/authors: LREC 2026 (arXiv preprint) / Hend Al-Khalifa and 7 co-authors (2026)
- Primary source: https://arxiv.org/abs/2602.13870
- Provenance status: `VERIFIED_LEVEL_1_2 (arXiv preprint + LREC 2026 acceptance; lead author is an established Arabic-NLP researcher)`

## Language and platform
- Languages: Arabic
- Dialects: MSA, Gulf, Egyptian, Levantine, Maghrebi
- Egypt-specific: PARTIAL (one of four dialect groups covered, exact Egyptian proportion not stated in available sources)
- Platforms: social media, e-commerce, customer service, a fourth unspecified platform

## Domain, labels, size
- Tasks: politeness classification
- Labels: polite, impolite, neutral, 16 finer politeness categories
- Sample count reported: 10000
- Sample count independently verified: None

## License and commercial-use status
- License reported: Creative Commons Attribution 4.0 (CC BY 4.0) -- confirmed directly from the arXiv paper page this session
- License verified: True
- Commercial-use status: `LIKELY_ALLOWED_REVIEW_REQUIRED (CC BY 4.0 permits commercial use with attribution; standard due-diligence review still recommended before integration)`
- Redistribution status: Permitted under CC BY 4.0 with attribution

## Quality findings
- Schema status: NOT_INSPECTED
- Duplicate status: NOT_MEASURED
- Language audit: NOT_PERFORMED

## PII findings
- PII risk: UNKNOWN

## Known limitations
Only download location found (huggingface.co/datasets/gagan3012/adab) is an unofficial-looking third-party account, not confirmed to be author-controlled or an authorized mirror. Per governing instructions, a LEVEL_5/6 mirror is not used to substitute for primary-source confirmation. NOT downloaded this session pending confirmation of an official release channel.

## Recommended role
unspecified

## Approval status
`ACCESS_PENDING` (Tier `TIER_A_CORE`)

## Review notes
HIGHEST-PRIORITY new discovery this session: recent (2026), clear CC BY 4.0 license verified from the primary paper, explicit e-commerce/customer-service domain coverage, includes Egyptian dialect, credible institutional authorship. Strongly recommended as the lead target for the next targeted-source-recovery gate to locate the official release channel before acquisition.

## Citation / evidence sources
- https://arxiv.org/abs/2602.13870 (fetched directly)
- https://aclanthology.org/2026.lrec-1.244/
- huggingface.co/datasets/gagan3012/adab (found, not used as authoritative)
