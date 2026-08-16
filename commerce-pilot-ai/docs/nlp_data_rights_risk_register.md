# NLP Data Rights / Legal Risk Register

Status: engineering risk classification only. This is **not** legal advice. Conservative classifications are used throughout; a qualified legal review is required before any commercial use of any dataset listed here.

| Dataset | License | Source-platform terms | Copyright risk | Redistribution risk | Commercial-use risk | Scraping-origin risk | Personal-data risk | Provenance | Author-contact required? |
|---|---|---|---|---|---|---|---|---|---|
| Amazon Appliances | Not stated on verified project page | Amazon.com ToS not independently reviewed | MEDIUM | UNKNOWN | UNKNOWN | LOW (academic aggregator, McAuley Lab) | MEDIUM | VERIFIED_LEVEL_1 | No (already acquired under existing project approval) |
| LABR | GPLv2 (verified from LICENSE file) | Goodreads ToS not independently reviewed | MEDIUM | Permitted under GPLv2 (share-alike) | REVIEW_REQUIRED (copyleft obligations) | MEDIUM (author-run scrape of Goodreads) | LOW-MEDIUM | VERIFIED_LEVEL_2 | No |
| ASTD | GPLv2 (verified from LICENSE file) | X/Twitter ToS not independently reviewed | MEDIUM | Permitted under GPLv2 (share-alike) | REVIEW_REQUIRED (copyleft obligations) | MEDIUM (author-run Twitter collection) | LOW-MEDIUM | VERIFIED_LEVEL_2 | No |
| MPOLD | Apache 2.0 at repository level (verified from LICENSE file) | Twitter/Facebook/YouTube ToS not independently reviewed; comment text originates from third-party platform posts distinct from the repo's own code license | MEDIUM (repo license may not fully cover underlying third-party comment rights) | LIKELY_ALLOWED_REVIEW_REQUIRED | LIKELY_ALLOWED_REVIEW_REQUIRED | MEDIUM (multi-platform scrape) | MEDIUM (pending content audit) | VERIFIED_LEVEL_2 | No |
| Egyptian Tweets 40K | UNKNOWN (Harvard Dataverse page inaccessible) | Twitter ToS not reviewed | UNKNOWN | UNKNOWN | UNKNOWN | MEDIUM | UNKNOWN | VERIFIED_LEVEL_3 (DOI) | Possibly, if page access remains blocked |
| AEC2 (10K) | UNKNOWN | Twitter ToS not reviewed | UNKNOWN | UNKNOWN | UNKNOWN | MEDIUM | UNKNOWN | VERIFIED_LEVEL_2 (citation only) | Likely yes (no repository located) |
| ArSAS | UNKNOWN | Twitter ToS not reviewed | UNKNOWN | UNKNOWN | UNKNOWN | MEDIUM | UNKNOWN | VERIFIED_LEVEL_2 | Likely yes |
| ADAB | CC BY 4.0 (verified from arXiv paper) | Mixed (social media + e-commerce + customer service source platforms, terms not individually reviewed) | LOW (CC BY 4.0 is clear and permissive) | Permitted with attribution | LIKELY_ALLOWED_REVIEW_REQUIRED | MEDIUM (multi-platform) | MEDIUM (customer-service/e-commerce text) | VERIFIED_LEVEL_1_2 (paper); download channel unconfirmed (LEVEL_5 mirror only) | Yes — recommended, to confirm official release channel before acquisition |
| HARD | NO_LICENSE_FOUND | Booking.com ToS not reviewed (commercial travel platform) | HIGH (no repo license + commercial source platform) | UNVERIFIED | PROHIBITED_PENDING_REVIEW | MEDIUM-HIGH (scrape of a commercial platform) | MEDIUM | VERIFIED_LEVEL_2 | Recommended |
| Jumia Egypt Reviews (JERD) | UNKNOWN (Kaggle page inaccessible) | Jumia ToS not reviewed; uploader is not Jumia itself | HIGH (unverified third-party redistribution of e-commerce platform content) | UNKNOWN | UNKNOWN | HIGH (unknown collection method) | MEDIUM | VERIFIED_LEVEL_5 | Recommended (contact uploader or seek an official Jumia data channel) |
| Commercial call-center vendors | Paid commercial license (per vendor) | Vendor-specific contractual terms | LOW (if properly licensed) | Per contract | Per contract, likely ALLOWED if purchased | LOW (vendor-collected, presumably consented) | Requires vendor consent documentation review | VENDOR_ADVERTISED_NOT_AUDITED | N/A (commercial negotiation, not academic contact) |

## Cross-cutting notes

- **Code license vs. data license** is explicitly distinguished throughout: a repository's LICENSE file (e.g. MPOLD's Apache 2.0) governs the authors' compilation/annotation work; it does not automatically extend full rights over third-party-authored source text (tweets, comments, reviews) collected from social platforms.
- **GitHub availability is never treated as unrestricted rights** — ASTD and LABR both required a direct LICENSE-file fetch this session to establish GPLv2; before that fetch, an earlier secondary-source summary had incorrectly reported ASTD as having no stated license.
- **Kaggle uploads by individuals** (JERD) carry the highest redistribution-risk classification in this register, since the uploader is not the platform owner and no license could be verified.
