# CommercePilot AI — An Intelligent E-Commerce Decision Platform

> **Status note**: this file documents the project's original Phase 1
> foundation and is preserved as historical record — it is not rewritten
> retroactively. The project has since progressed well past this phase:
> trained models, a FastAPI service, and a live Medusa integration exist
> and are documented in the top-level
> [`README.md`](../README.md) under the public brand **HASEBHA | حاسبها**.
> See that file for the current, accurate project state.

## Project overview

CommercePilot AI is a decision-support platform for e-commerce teams. It is intended to turn operational, behavioral, and customer-feedback signals into clear, reviewable recommendations that help teams decide what to investigate and what action to take.

This repository contains the project foundation, Phase 1A acquisition/validation tooling, Phase 1B independent cleaning and quality assessment, and the Phase 1C Olist modeling-readiness audit and experiment specification. It does not contain a final feature matrix, trained models, services, or user interfaces.

Phase 1C assigned Olist a technical **NO-GO**. Phase 1D reassessed a narrower, retrospective stored-estimate benchmark as **CONDITIONAL GO** using timing-only strict-core features. Legal/licensing, production, and Egyptian-market external-validity gates remain **NO-GO**. No model or feature matrix has been created.

## Core business problem

E-commerce decisions are often made from disconnected operational reports, customer behavior, and review feedback. Teams need a consistent way to identify meaningful signals, understand why they matter, assess supporting evidence, and translate them into practical next actions without obscuring uncertainty or human accountability.

## Planned AI capabilities

- **Fulfillment and delivery intelligence:** identify delivery performance patterns and operational issues worth investigating.
- **Product recommendations and personalization:** support relevant product discovery based on customer behavior.
- **Voice-of-customer intelligence:** detect sentiment, recurring issues, and themes in customer reviews.
- **Admin decision dashboard:** present evidence and recommended actions through a consistent decision-support experience.

## High-level architecture

Each capability will have an independent data and analysis path. Future data pipelines will prepare capability-specific inputs, and future AI services will produce a shared **Decision Action Card** contract. An admin dashboard may later consume that contract to display findings, evidence, confidence or limitations, and suggested actions.

```text
Independent dataset
        |
Capability-specific data pipeline
        |
Capability-specific analysis or AI service
        |
Shared Decision Action Card API contract
        |
Admin decision dashboard
```

The shared contract is an integration boundary for decision outputs, not a reason to combine the underlying datasets.

## Planned tech stack

The stack will be selected and validated during later phases. Current planning anticipates:

- Python for data processing, analysis, and AI services
- YAML for environment-independent configuration templates
- Notebook tooling for controlled exploration and research
- Automated tests for data, analytical, and contract behavior
- A separately evaluated API and admin dashboard stack
- Version control and reproducible dependency management

No framework or infrastructure choice is established by this foundation phase.

## Project phases

1. **Foundation:** define the charter, research questions, data governance, risks, and repository layout.
2. **Data validation:** verify licensing and sources, acquire approved datasets, profile quality, and document capability-specific data contracts.
3. **Capability research:** establish baselines, evaluation methods, and evidence for each independent capability.
4. **Decision contract:** design and validate the shared Decision Action Card API contract.
5. **Product integration:** implement services and an admin decision dashboard around validated capabilities.
6. **Operational readiness:** address monitoring, privacy, deployment, adoption, and ongoing validation.

## Dataset independence

The Olist, Instacart, and Amazon Reviews 2023 (Appliances only) datasets will remain independent. They must not be merged. Each supports a separate AI capability and may connect to the wider platform only by producing outputs that conform to the future shared Decision Action Card API contract.

## Committee Package (2026-08-22)

The academic/committee defense package (D1-D6, evidence-traced against repository artifacts)
lives under `docs/committee/`:

- `D1_ACADEMIC_DOCUMENT.md` — full academic write-up (Abstract → Conclusion → Appendices)
- `D2_DEFENSE_SLIDES.md` — 22-slide defense deck (structured markdown, speaker notes)
- `D3_RESULTS_TABLE_WITH_CI.md` / `.json` — unified results table with bootstrap confidence intervals
- `D4_ARTIFACT_INDEX.md` — full path-traced index of every artifact referenced
- `D5_COMMITTEE_QA_BRIEF.md` — 15 hardest anticipated questions with evidence-backed answers
- `D6_DEMO_SCRIPT.md` — 3-5 minute live demo script with fallback paths

See also `docs/DECISIONS_FINAL.md` (final-advisor decision log) and
`docs/HASEBHA_SLA_BUSINESS_DECISION_MEMO.md` (the one remaining business decision).

## Notebooks

- `notebooks/01_olist_eda_and_model_audit.ipynb` — Olist EDA and model audit
- `notebooks/02_arabic_nlp_eda_and_analysis.ipynb` — Arabic NLP EDA
- `notebooks/03_amazon_reviews_eda_and_analysis.ipynb` — Amazon reviews EDA
- `notebooks/04_amazon_sentiment_modeling.ipynb` — Amazon sentiment modeling
- `notebooks/05_arabic_foundation_sentiment_modeling.ipynb` — Arabic foundation sentiment modeling
- `notebooks/06_transfer_gap_and_funnel.ipynb` — live, artifact-grounded committee demo: the
  research-to-production transfer-gap ablation (0.7686 → 0.5540, 94% seller-history
  attribution) and the Marketing Funnel enrichment negative result (NR-16). Runs end-to-end in
  under 20 seconds; every displayed number is read from a saved artifact or recomputed live
  and cross-checked against it in the notebook itself.
