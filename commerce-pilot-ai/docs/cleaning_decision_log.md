# Cleaning Decision Log

## Olist

| Observed issue | Evidence | Decision and rationale | Effect | Code | Reversible |
|---|---|---|---|---|---|
| Empty CSV fields and mixed physical/timestamp types | Nine source CSVs; null counts in summary | Parse empty fields as null and infer types using the full file; preserve source table boundaries. | 0 rows/fields intentionally dropped. | `clean_olist.py` | Yes, raw CSVs are immutable. |
| Missing order/product/review fields | Orders, products, reviews | Retain nulls; meaning depends on lifecycle/content availability. | Missing values remain explicit. | `clean_olist.py` | Yes |
| Timestamp sequence anomalies | Orders: 1,359 carrier-before-approval; 23 delivery-before-carrier | Retain and document; do not guess corrected times. | No exclusions. | `phase1b_observations.py` | Yes |

## Instacart

| Observed issue | Evidence | Decision and rationale | Effect | Code | Reversible |
|---|---|---|---|---|---|
| Empty prior-order interval | `orders.csv`: 206,209 nulls | Preserve as null; imputation could misrepresent first-order semantics. | 0 rows dropped. | `clean_instacart.py` | Yes |
| Large relational CSVs | Six CSV files | Convert independently to typed, compressed Parquet without joins. | All rows/fields retained. | `clean_instacart.py` | Yes |
| Provenance limitations | `data_provenance.md` | Do not alter Not verified origin/version/checksum/license states. | Blocks modeling readiness. | Documentation gate | Yes |

## Amazon Appliances

| Observed issue | Evidence | Decision and rationale | Effect | Code | Reversible |
|---|---|---|---|---|---|
| 2,065 empty/whitespace texts | `Appliances.jsonl.gz` | Retain and flag with `has_usable_text`; avoid silent deletion. | 0 rows dropped; one quality field added. | `clean_amazon_appliances.py` | Yes |
| 22,657 duplicate candidate identities | Five-field candidate identity | Retain until review semantics and duplicate policy are approved. | Duplicates remain measurable. | `data_quality.py` | Yes |
| Epoch-millisecond timestamp | Observed `timestamp` integer | Add `review_datetime_utc` using `epoch_ms`; preserve original integer. | One derived parsing field; original retained. | `clean_amazon_appliances.py` | Yes |
| No sentiment/language label | Observed schema | Do not infer or synthesize either. | No labels added. | Cleaner and readiness docs | Yes |

