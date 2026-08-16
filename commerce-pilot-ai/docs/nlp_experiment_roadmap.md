# NLP Experiment Roadmap (design only — no experiment executed)

Status: planning document. No experiment below has been run. Each defines what a future experiment would prove and would **not** prove, per the governing session's requirement.

## Experiment A — English e-commerce review benchmark

- **Data:** Amazon Appliances (`APPROVED_FOR_BENCHMARK_ONLY`).
- **Would prove:** whether standard sentiment/aspect/complaint-methodology baselines work on English e-commerce review text at all, and establish a reusable pipeline shape.
- **Would NOT prove:** anything about Arabic, Egyptian Arabic, code-switching, or Egyptian-market performance.

## Experiment B — Egyptian sentiment benchmark

- **Data:** Egyptian Tweets 40K (pending access recovery), AEC2 10K (pending access recovery), ASTD (acquired, Egypt-specificity unconfirmed).
- **Would prove:** whether a sentiment classifier trained/evaluated on genuinely Egyptian-labeled text achieves reasonable performance, and whether methodology from Experiment A transfers to Arabic script.
- **Would NOT prove:** e-commerce-domain performance (these corpora are general-topic Twitter, not commerce text) or code-switching robustness.

## Experiment C — Egyptian code-switch benchmark

- **Data:** none currently acquired at sufficient scale with sentiment/complaint labels. ArzEn (speech, ASR-focused, not sentiment-labeled) is the only genuinely Egyptian code-switch resource found this session.
- **Would prove:** (once suitable labeled data exists) whether a model handles Arabic-English/Franco-Arabic code-switching in customer text.
- **Would NOT prove:** anything today — this experiment is currently **blocked on data availability**, not merely unexecuted.

## Experiment D — Arabic multi-domain robustness

- **Data:** LABR (books), HARD (hotels, quarantined pending license review), ADAB (multi-domain including e-commerce, access pending) as distinct domains.
- **Would prove:** whether a model trained on one Arabic domain generalizes to another (e.g. book-review sentiment → hotel-review sentiment), a standard domain-transfer robustness check.
- **Would NOT prove:** Egyptian dialect robustness specifically (LABR/HARD are pan-Arab, not Egypt-labeled) or e-commerce-domain performance beyond ADAB's partial coverage.

## Experiment E — Complaint/intent benchmark

- **Data:** **none currently available.** No dataset in the investigated portfolio has `COMPLAINT` or `INTENT` labels as first-class categories (`docs/nlp_label_ontology_mapping.md`).
- **Would prove:** nothing yet — this experiment is blocked on data availability. The closest raw-text-potential source (Arabic food-delivery sentiment research, access pending) would need new complaint/intent annotation before this experiment is possible.

## Experiment F — Multi-task / shared-representation research

- **Scientific justification required before definition:** whether a shared representation across sentiment (Experiment B), politeness (ADAB), and offensive-language (MPOLD) tasks provides efficiency or robustness gains is a legitimate research question, but is **not yet justified** as a near-term priority given that Experiments B-E are still data-blocked. Deferred until at least two of B/C/D/E have real executed baselines to compare against.

## Experiment G — Future first-party Egyptian commerce validation

- **Data:** none exists (`docs/nlp_egyptian_first_party_data_gap_analysis.md`).
- **Would prove:** actual Egyptian-market operational validity — the only experiment in this roadmap that could support an Egyptian-market-readiness claim.
- **Would NOT be possible** until first-party data collection (WhatsApp/chat/call-center/return-reason streams) is designed, authorized, and executed under a documented consent basis.

## Sequencing note

Experiments A and D can begin (data-wise) once a future NLP execution gate authorizes them; A uses only already-approved Amazon Appliances data, D uses LABR (approved) plus HARD (pending a license review) plus ADAB (pending access recovery). Experiments B, C, E, and G are data-blocked and require the recovery/acquisition work identified in this session's registry before they are executable at all. No experiment is executed in this session.
