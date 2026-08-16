# Amazon Appliances Voice-of-Customer Readiness

The physical source provides `title`, `text`, and `rating` (1-5), plus image references, product identifiers, pseudonymous user ID, epoch-millisecond timestamp, helpful-vote count, and verified-purchase flag. The active NLP adapter maps these explicitly to canonical `review_title`, `review_text`, and `overall`; the physical source itself does not contain those canonical names. It does not provide explicit sentiment, topic, complaint, language, geography, or operational resolution labels.

All 2,128,605 parsed records are retained. The cleaner flags 2,065 empty/whitespace texts rather than silently deleting them and derives a timezone-neutral UTC datetime. It reports 22,657 duplicate candidate review identities but retains them pending a documented duplicate policy. Malformed JSON causes the pipeline to fail rather than skip records; zero parsing failures occurred in the completed run.

Language remains unknown. Future work must define a language-identification and annotation policy before text modeling. Reports must use record references rather than copying customer text, and user IDs/image links should be excluded unless essential. License terms remain Not verified.

Future Decision Action Cards could contain a recurring complaint theme, affected product reference, privacy-reviewed evidence references, suggested owner/action, confidence, and limitations. Theme and sentiment would be inferred future outputs, not observed labels.

Amazon Appliances reviews are not a substitute for Arabic reviews from an Egyptian live store. Production readiness requires consented local feedback, Arabic/dialect evaluation, store catalog linkage, support outcomes, and current policy context. No NLP or sentiment model was created.
