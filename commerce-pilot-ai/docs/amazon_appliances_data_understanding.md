# Amazon Appliances Data Understanding

## Scope and schema

The inspected file is `data/raw/amazon_reviews_appliances/Appliances.jsonl.gz`, containing 2,128,605 parsed JSON Lines records. Observed fields and inferred types are:

| Field | Type | Nulls |
|---|---|---:|
| `rating` | double | 0 |
| `title`, `text`, `asin`, `parent_asin`, `user_id` | string | 0 each |
| `images` | list of image-reference structures | 0 |
| `timestamp`, `helpful_vote` | integer | 0 each |
| `verified_purchase` | boolean | 0 |

The physical candidate identity `(user_id, parent_asin, timestamp, rating, text)` has 22,657 duplicate rows beyond its distinct count. These were measured but retained. There are 2,065 empty or whitespace-only review texts and 2,126,540 usable texts.

For active NLP execution only, `src/nlp/amazon_adapter.py` explicitly maps physical `rating`, `title`, and `text` to canonical `overall`, `review_title`, and `review_text`. The adapter fails when a required physical field is absent or a canonical field is already present; it has no fallback alias selection.

## Observed coverage and distributions

- Derived UTC timestamp coverage: `2000-10-23 14:37:23` to `2023-09-12 16:29:17.016`.
- Ratings: 1 = 250,453; 2 = 79,664; 3 = 104,047; 4 = 208,216; 5 = 1,486,225.
- Helpful votes range from 0 to 5,704; no negative value was observed.
- Verified purchase: true 2,040,064; false 88,541.

## Quality, privacy, and transfer limitations

No explicit sentiment, issue, topic, language, geography, or Egyptian-store field was observed. Ratings and text are observations; sentiment must not be asserted from either without a defined future labeling protocol. Language is unknown because language detection was not performed. Review text, user IDs, product IDs, and image references are minimized to controlled processed storage and must not be exposed in reports.

The category is Appliances only. Amazon customers, catalog, review behavior, languages, and marketplace context are not substitutes for Arabic reviews, Egyptian products, local policies, or consented live-store feedback.
