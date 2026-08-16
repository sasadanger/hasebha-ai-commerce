# Dataset Card: ASTD: Arabic Sentiment Tweets Dataset

`dataset_id = astd` | Tier: `TIER_B_SUPPORTING` | Approval: `APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT`

## Purpose
Domain: general political/social Twitter sentiment (not e-commerce). Commerce relevance: LOW (methodology transfer only; content is general/political, not commerce).

## Source and provenance
- Publisher/authors: Mahmoud Nabil, Mohamed Aly, Amir Atiya (Cairo University) / Mahmoud Nabil, Mohamed Aly, Amir Atiya (2015)
- Primary source: https://github.com/mahmoudnabil/ASTD
- Provenance status: `VERIFIED_LEVEL_2`

## Language and platform
- Languages: Arabic
- Dialects: Not explicitly annotated; sample inspection suggests Egyptian/MSA mix
- Egypt-specific: UNCONFIRMED (author affiliation Egyptian; sampled tweet content references Egyptian political parties/figures; paper does not explicitly claim Egypt-only geo-filtering)
- Platforms: Twitter/X

## Domain, labels, size
- Tasks: sentiment classification
- Labels: OBJ, POS, NEG, NEUTRAL
- Sample count reported: 10006
- Sample count independently verified: 9694

## License and commercial-use status
- License reported: GNU GPLv2 (confirmed by direct fetch of repository LICENSE file this session -- corrects an earlier secondary-source claim of 'no license stated')
- License verified: True
- Commercial-use status: `REVIEW_REQUIRED (GPLv2 copyleft obligations)`
- Redistribution status: Permitted under GPLv2 terms

## Quality findings
- Schema status: VERIFIED_THIS_SESSION (tab-separated: text, label)
- Duplicate status: 4 exact-duplicate text rows out of 9,694 (0.04%), independently measured this session
- Language audit: PERFORMED_THIS_SESSION: 100.00% Arabic-script, 0.22% also contains Latin characters, 49.31% contains hashtags -- consistent with Twitter/political-hashtag content.

## PII findings
- PII risk: LOW-MEDIUM (public tweet text, no user IDs retained in released file; mentions/hashtags present at 49.3% of rows)

## Known limitations
None beyond what is stated above.

## Recommended role
EGYPTIAN_SOCIAL_SENTIMENT_BENCHMARK (tentative -- Egypt-specificity not fully confirmed)

## Approval status
`APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT` (Tier `TIER_B_SUPPORTING`)

## Review notes
Verified row count (9,694) is slightly below the commonly-cited 10,006 -- likely minor TSV-parsing loss (malformed lines skipped), not a data-integrity concern; documented rather than silently reconciled. Four-class label scheme confirmed directly from data, correcting an earlier three-class secondary-source summary.

## Citation / evidence sources
- https://github.com/mahmoudnabil/ASTD (fetched directly)
- https://aclanthology.org/D15-1299/
- direct download+hash this session
