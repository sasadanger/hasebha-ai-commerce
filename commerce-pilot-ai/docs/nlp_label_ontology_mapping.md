# NLP Label Ontology Mapping

Status: planning/registry document. Datasets are **not** forced into a single shared label set. Each task family below is kept distinct. Mappings are recorded as `EXACT_MAPPING`, `PARTIAL_MAPPING`, or `NO_MAPPING` — never inferred silently.

## Task families (kept separate)

`SENTIMENT`, `EMOTION`, `COMPLAINT`, `SPEECH_ACT`, `INTENT`, `OFFENSIVE_LANGUAGE`, `TOXICITY`, `POLITENESS`, `ASPECT`, `TOPIC`, `RATING`.

## Source label → conceptual category mapping

| Dataset | Source label(s) | Task family | Mapping type | Rationale |
|---|---|---|---|---|
| Amazon Appliances | `overall` (1-5, observed) | `RATING` | `EXACT_MAPPING` | Direct numeric score; not a sentiment label. |
| LABR | `rating` (1-5, observed) | `RATING` | `EXACT_MAPPING` | Same as above — a rating is not automatically a sentiment label. |
| ASTD | `OBJ`/`POS`/`NEG`/`NEUTRAL` | `SENTIMENT` | `EXACT_MAPPING` | Direct sentiment annotation by original authors. |
| Egyptian Tweets 40K / AEC2 (10K) | `positive`/`negative` | `SENTIMENT` | `EXACT_MAPPING` | Direct binary sentiment annotation. |
| ArSAS | 4 sentiment classes + 6 speech-act classes | `SENTIMENT` + `SPEECH_ACT` | `EXACT_MAPPING` (both, kept as two separate label sets) | The corpus explicitly separates sentiment from speech-act; they are not merged into one label. |
| MPOLD | `Non-Offensive`/`Offensive` (+HS/V/other) | `OFFENSIVE_LANGUAGE` | `EXACT_MAPPING` | Explicitly an offensive-language annotation, not a sentiment or complaint label. |
| EmotionalTone | joy/anger/sympathy/sadness/fear/surprise/love/none | `EMOTION` | `EXACT_MAPPING` | Direct emotion annotation, 8 classes. |
| ADAB | polite/impolite/neutral + 16 sub-categories | `POLITENESS` | `EXACT_MAPPING` | A dedicated politeness ontology; explicitly not treated as sentiment. |
| HARD | `rating` (1-5, observed) | `RATING` | `EXACT_MAPPING` | Same rating-vs-sentiment distinction as above. |

## Explicitly forbidden mappings

Per governing instructions, the following collapses are **never** performed in this registry or in any future experiment built on it:

- `offensive = negative sentiment` — MPOLD's offensive/non-offensive label is never treated as a sentiment polarity label.
- `1-star = complaint` — no rating-based dataset (Amazon Appliances, LABR, HARD) has a `COMPLAINT` label; a 1-star rating is evidence a complaint task *could* be built on top of the raw text, but it is not itself a complaint annotation. All three are marked `RAW_TEXT_POTENTIAL`, not `ANNOTATED_SUPPORT`, for the `COMPLAINT` category in `docs/nlp_domain_coverage_matrix.md`.
- `politeness = sentiment` — ADAB's polite/impolite/neutral labels are not remapped onto positive/negative sentiment; politeness and sentiment are orthogonal constructs (an impolite message can be positive in content, and vice versa).
- `emotion = sentiment` — EmotionalTone's 8-way emotion label is not collapsed into a 2- or 3-way sentiment polarity.

## No dataset in this registry currently has

`COMPLAINT`, `INTENT`, `ASPECT`, or `TOPIC` labels as first-class annotated categories. These remain `NO_MAPPING` / gaps across the entire investigated portfolio — see `docs/nlp_egyptian_first_party_data_gap_analysis.md` for what would be required to close them.
