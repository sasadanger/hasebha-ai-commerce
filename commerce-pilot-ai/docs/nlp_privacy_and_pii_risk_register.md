# NLP Privacy and PII Risk Register

Status: planning/registry document. No deanonymization was attempted. No user was enriched with external identity information. This register classifies risk only; it does not resolve it.

| Dataset | Usernames | Names | Emails | Phones | Addresses | Order IDs | Location | Social handles | URLs | Personal narratives | Sensitive attributes | Overall PII risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon Appliances | Pseudonymous `user_id` present | Possible in free text | Not observed | Not observed | Not observed | N/A | Not observed | Not observed | Not observed | Yes (review narratives) | Possible (health/medical mentions in appliance reviews cannot be ruled out) | MEDIUM |
| LABR | Pseudonymous `user_id` present | Possible in free text | Not inspected | Not inspected | Not observed | N/A | Not observed | Not observed | Not inspected | Yes (review narratives) | Not assessed | LOW-MEDIUM |
| ASTD | Not retained in released file | Possible in free text | Not observed | Not observed | Not observed | N/A | Not observed | Mentions/hashtags at 49.3% of rows (measured this session) | Not observed | Yes (political opinion is itself a sensitive attribute) | Political opinion (sensitive) | LOW-MEDIUM |
| MPOLD | Not inspected (xlsx audit deferred) | Not inspected | Not inspected | Not inspected | Not inspected | N/A | Not inspected | Not inspected | Not inspected | Yes (news comments) | Political/social commentary likely (deferred confirmation) | MEDIUM (pending full audit) |
| Egyptian Tweets 40K / AEC2 | Not obtained | Not obtained | Not obtained | Not obtained | Not obtained | N/A | Not obtained | Not obtained | Not obtained | Likely (tweets) | Not assessed | UNKNOWN (data not obtained) |
| ADAB | Not obtained | Not obtained | Not obtained | Not obtained | Not obtained | N/A | Not obtained | Not obtained | Not obtained | Likely (customer-service/e-commerce text) | Not assessed | UNKNOWN (data not obtained) |
| HARD | Not obtained | Not obtained | Not obtained | Not obtained | Not obtained | N/A | Not obtained | Not obtained | Not obtained | Likely (hotel reviews) | Not assessed | UNKNOWN (data not obtained) |
| Jumia Egypt Reviews (JERD) | Not obtained | Not obtained | Not obtained | Not obtained | Not obtained | Possible (order-linked reviews plausible) | Possible | Not obtained | Not obtained | Likely | Not assessed | UNKNOWN (data not obtained) |
| Commercial call-center vendors | Speaker identity per vendor terms | Possible | Possible | Likely (call metadata) | Possible | Possible | Likely | N/A | N/A | Yes (full conversations) | Likely (health/financial mentions possible in retail/telecom/healthcare verticals) | HIGH (full conversational speech data) |

## General principles applied

- No dataset in this registry is treated as PII-cleared merely because it is publicly released.
- Pseudonymous identifiers (`user_id`, `parent_asin`, etc.) are retained only in controlled processed storage per existing project practice (`docs/amazon_appliances_data_understanding.md`); this register extends the same discipline to all newly investigated candidates.
- Free-text review/tweet/comment fields are treated as carrying residual narrative-PII risk by default, regardless of platform, until a dedicated redaction/minimization pass is designed.
- Commercial call-center speech vendors carry the highest risk in this register because they involve full conversational audio with real customers, not anonymized/aggregated review text — any future engagement with such a vendor requires a dedicated legal/privacy review before any data is received, not merely at ingestion time.

No repository code or process in this session accessed, redacted, or persisted any PII beyond what already existed in previously-approved Amazon Appliances processed storage.
