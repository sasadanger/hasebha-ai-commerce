# Dataset Card: LABR: Large-Scale Arabic Book Reviews Dataset

`dataset_id = labr` | Tier: `TIER_B_SUPPORTING` | Approval: `APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT`

## Purpose
Domain: book reviews (general product-review methodology transfer only, not e-commerce/delivery). Commerce relevance: MEDIUM (review-sentiment methodology transfers; domain is books, not e-commerce/logistics).

## Source and provenance
- Publisher/authors: Mohamed Aly, Amir Atiya (Cairo University) / Mohamed Aly, Amir Atiya (2013)
- Primary source: https://github.com/mohamedadaly/LABR
- Provenance status: `VERIFIED_LEVEL_2`

## Language and platform
- Languages: Arabic
- Dialects: Mixed MSA and various dialects, not individually labeled
- Egypt-specific: False
- Platforms: Goodreads

## Domain, labels, size
- Tasks: sentiment (1-5 star rating as proxy)
- Labels: rating (1-5)
- Sample count reported: 63257
- Sample count independently verified: 63257

## License and commercial-use status
- License reported: GNU GPLv2 (confirmed by direct fetch of repository LICENSE file this session)
- License verified: True
- Commercial-use status: `REVIEW_REQUIRED (GPLv2 permits use but imposes copyleft/share-alike obligations on derivative works; not a simple commercial-clear license)`
- Redistribution status: Permitted under GPLv2 terms (share-alike)

## Quality findings
- Schema status: VERIFIED_THIS_SESSION (rating, review_id, user_id, book_id, text)
- Duplicate status: 3,167 exact-duplicate text rows out of 63,257 (5.0%), independently measured this session
- Language audit: PERFORMED_THIS_SESSION: 100.00% of text contains Arabic-script characters, 0.00% Latin, in full-corpus scan.

## PII findings
- PII risk: LOW-MEDIUM (pseudonymous Goodreads user_id; free text may contain incidental personal opinion, not typically PII)

## Known limitations
None beyond what is stated above.

## Recommended role
ARABIC_REVIEW_BENCHMARK

## Approval status
`APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT` (Tier `TIER_B_SUPPORTING`)

## Review notes
Real data downloaded and independently verified this session (63,257 rows matches the widely-cited ~63,000 figure). General Arabic, NOT Egyptian-specific -- must not be relabeled Egyptian per governing instructions. Strong provenance (author's own GitHub, clear GPLv2 license from LICENSE file).

## Citation / evidence sources
- https://github.com/mohamedadaly/LABR (fetched directly)
- https://aclanthology.org/P13-2088/
- direct download+hash this session
