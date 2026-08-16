# NLP Platform Coverage Matrix

Coverage marked only where directly evidenced. Blank/False = not evidenced, not fabricated.

## Platform coverage

| Dataset | amazon_marketplace_reviews | x_twitter | facebook | youtube | google_play_apps | customer_service | delivery_logistics | whatsapp | calls_transcripts | product_reviews | complaints | refund_intent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon Reviews 2023 -- Appliances | True | False | False | False | False | False | False | False | False | True | False | False |
| LABR: Large-Scale Arabic Book Reviews Dataset | False | False | False | False | False | False | False | False | False | True | False | False |
| ASTD: Arabic Sentiment Tweets Dataset | False | True | False | False | False | False | False | False | False | False | False | False |
| Multi-Platform Arabic Offensive Language Dataset (news comments) | False | True | True | True | False | False | False | False | False | False | False | False |
| Corpus on Arabic Egyptian Tweets (40K) | False | True | False | False | False | False | False | False | False | False | False | False |
| Arabic Egyptian Corpus 2 (AEC2, 10K) | False | True | False | False | False | False | False | False | False | False | False | False |
| ArSAS: Arabic Speech-Act and Sentiment Corpus of Tweets | False | True | False | False | False | False | False | False | False | False | False | False |
| ArzEn: A Speech Corpus for Code-switched Egyptian Arabic-English | False | False | False | False | False | False | False | False | True | False | False | False |
| Emotional Tone Detection in Arabic Tweets (EmotionalTone) | False | True | False | False | False | False | False | False | False | False | False | False |
| Arabic Sentiment Analysis of Food Delivery Services Reviews (research) | False | False | False | False | False | False | True | False | False | False | True | False |
| ADAB: Arabic Dataset for Automated Politeness Benchmarking | False | False | False | False | False | True | False | False | False | True | False | False |
| HARD: Hotel Arabic-Reviews Dataset | False | False | False | False | False | False | False | False | False | True | False | False |
| Jumia Egypt Reviews Dataset (JERD) | False | False | False | False | False | False | False | False | False | True | False | False |
| Commercial Egyptian Arabic call-center/customer-service speech data vendors (FutureBeeAI, Macgence, and similar) | False | False | False | False | False | True | True | False | True | False | False | False |
| Arabic Inquiry-Answer Dialogue Acts corpus (Egyptian banks / EgyptAir calls) | False | False | False | False | False | True | False | False | True | False | False | False |

## Language coverage

| Dataset | egyptian_arabic | msa | english | code_switch | franco_arabic |
|---|---|---|---|---|---|
| Amazon Reviews 2023 -- Appliances | False | False | True | False | False |
| LABR: Large-Scale Arabic Book Reviews Dataset | False | True | False | False | False |
| ASTD: Arabic Sentiment Tweets Dataset | likely_unconfirmed | True | False | False | False |
| Multi-Platform Arabic Offensive Language Dataset (news comments) | False | True | False | False | False |
| Corpus on Arabic Egyptian Tweets (40K) | True | False | False | False | False |
| Arabic Egyptian Corpus 2 (AEC2, 10K) | True | False | False | False | False |
| ArSAS: Arabic Speech-Act and Sentiment Corpus of Tweets | False | True | False | False | False |
| ArzEn: A Speech Corpus for Code-switched Egyptian Arabic-English | True | False | True | True | False |
| Emotional Tone Detection in Arabic Tweets (EmotionalTone) | True | False | False | False | False |
| ADAB: Arabic Dataset for Automated Politeness Benchmarking | True | True | False | False | False |
| HARD: Hotel Arabic-Reviews Dataset | False | True | False | False | False |
| Jumia Egypt Reviews Dataset (JERD) | unconfirmed | False | False | False | False |
| Commercial Egyptian Arabic call-center/customer-service speech data vendors (FutureBeeAI, Macgence, and similar) | True | False | False | False | False |
| Arabic Inquiry-Answer Dialogue Acts corpus (Egyptian banks / EgyptAir calls) | True | False | False | False | False |

## Task coverage

| Dataset | sentiment | emotion | speech_act | offensive_safety | intent | topics_aspects | politeness |
|---|---|---|---|---|---|---|---|
| Amazon Reviews 2023 -- Appliances | raw_rating_only | False | False | False | False | False | False |
| LABR: Large-Scale Arabic Book Reviews Dataset | raw_rating_only | False | False | False | False | False | False |
| ASTD: Arabic Sentiment Tweets Dataset | True | False | False | False | False | False | False |
| Multi-Platform Arabic Offensive Language Dataset (news comments) | False | False | False | True | False | False | False |
| Corpus on Arabic Egyptian Tweets (40K) | True | False | False | False | False | False | False |
| Arabic Egyptian Corpus 2 (AEC2, 10K) | True | False | False | False | False | False | False |
| ArSAS: Arabic Speech-Act and Sentiment Corpus of Tweets | True | False | True | False | False | False | False |
| Emotional Tone Detection in Arabic Tweets (EmotionalTone) | False | True | False | False | False | False | False |
| ADAB: Arabic Dataset for Automated Politeness Benchmarking | False | False | False | False | False | False | True |
| HARD: Hotel Arabic-Reviews Dataset | raw_rating_only | False | False | False | False | False | False |
