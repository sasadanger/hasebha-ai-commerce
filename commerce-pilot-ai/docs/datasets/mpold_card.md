# Dataset Card: Multi-Platform Arabic Offensive Language Dataset (news comments)

`dataset_id = mpold` | Tier: `TIER_C_RESEARCH_ONLY` | Approval: `APPROVED_FOR_BENCHMARK_ONLY`

## Purpose
Domain: offensive-language/safety, NOT customer sentiment. Commerce relevance: LOW.

## Source and provenance
- Publisher/authors: Qatar Computing Research Institute (QCRI) / Shammur Absar Chowdhury, Hamdy Mubarak, Ahmed Abdelali, Soon-gyo Jung, Bernard J. Jansen, Joni Salminen (2020)
- Primary source: https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset
- Provenance status: `VERIFIED_LEVEL_2`

## Language and platform
- Languages: Arabic
- Dialects: Multi-dialect, not individually broken out by country in available sources
- Egypt-specific: False
- Platforms: Twitter, Facebook, YouTube

## Domain, labels, size
- Tasks: offensive language detection, hate speech detection
- Labels: Non-Offensive, Offensive (subtypes: HS=hate speech, V=vulgar, other)
- Sample count reported: 4000
- Sample count independently verified: NOT_YET_COUNTED (xlsx schema inspection deferred -- openpyxl not installed in project environment; file downloaded and hashed)

## License and commercial-use status
- License reported: Apache License 2.0 (confirmed by direct fetch of repository LICENSE file this session)
- License verified: True
- Commercial-use status: `LIKELY_ALLOWED_REVIEW_REQUIRED (Apache 2.0 is permissive at the repository level, but underlying comment text originates from third-party social posts whose own platform terms are not independently verified -- code license vs. underlying content rights must be distinguished)`
- Redistribution status: Repository-level permitted under Apache 2.0; underlying third-party text rights unverified

## Quality findings
- Schema status: FILE_DOWNLOADED_SCHEMA_INSPECTION_DEFERRED (xlsx format; openpyxl dependency not installed, not added without separate authorization)
- Duplicate status: NOT_MEASURED (deferred)
- Language audit: NOT_PERFORMED (deferred)

## PII findings
- PII risk: MEDIUM (public social comments on news, potential for identifiable opinion/political content)

## Known limitations
Not customer-sentiment data; must not be mislabeled as complaint/sentiment benchmark. Content-level audit deferred pending an explicitly authorized environment dependency addition.

## Recommended role
SAFETY_TOXICITY_BENCHMARK, MULTIPLATFORM_ROBUSTNESS_BENCHMARK

## Approval status
`APPROVED_FOR_BENCHMARK_ONLY` (Tier `TIER_C_RESEARCH_ONLY`)

## Review notes
Explicitly NOT mislabeled as customer sentiment data per governing instructions -- this is a safety/offensive-language benchmark. Full content audit requires openpyxl; file is safely quarantined and hashed regardless.

## Citation / evidence sources
- https://github.com/shammur/Arabic-Offensive-Multi-Platform-SocialMedia-Comment-Dataset (fetched directly)
- https://aclanthology.org/2020.lrec-1.761.pdf
- direct download+hash this session
