# Modeling Readiness Gate

## Olist — Conditional Go after Phase 1D

- Provenance/raw validation/reproducible cleaning: complete; source version remains unverified and CC BY-NC-SA 4.0 suitability requires legal review.
- Prediction point: frozen at confirmed payment approval.
- Retrospective target formula: frozen as actual customer delivery later than recorded estimated delivery.
- Primary cohort: reproducible 95,082 unique orders; anomaly and sensitivity policies documented.
- Split: frozen chronological train/validation/test windows with 43,516 / 26,822 / 24,744 orders.
- Feature, aggregation, preprocessing, evaluation, and reproducibility contracts: complete.
- Blocking technical evidence: the repository does not prove the recorded estimated date is the immutable promise known at approval, or that conditional item/payment/catalog snapshots are complete at that time.
- Phase 1C verdict: **NO-GO** because approval-time estimate and snapshot semantics were unverified.
- Phase 1D evidence: future delivery is enforced as label-only; the claim is narrowed to retrospective stored-estimate exceedance; nine timing-only strict-core features avoid snapshot assumptions; business costs are separated from ranking evaluation.
- Technical offline gate: **CONDITIONAL GO** for the strict contract in a separate Phase 2A session, subject to legal/use authorization and exact contract enforcement.
- Legal/licensing, production, and Egyptian external-validity gates: **NO-GO**.
- See `phase1d_olist_readiness_reassessment.md`. Phase 1D did not begin modeling.

## Instacart — No-Go

- Raw validation and reproducible cleaning: complete.
- Provenance: archive origin, version, publisher checksum, and license remain Not verified.
- Target/evaluation event: conceptual held-out next-order event only.
- Split strategy: chronological per-user strategy documented; exact protocol not approved.
- Documentation: complete for Phase 1B.
- Gate: **No-Go** until provenance/license are resolved and evaluation/catalog requirements are approved.

## Amazon Appliances — No-Go

- Provenance/raw validation/reproducible cleaning: complete; license remains Not verified.
- Target definition: none; sentiment, issue, language, and topic labels are absent.
- Split strategy: must be defined after task/label policy, with temporal and product/user leakage controls.
- Documentation: complete for Phase 1B.
- Gate: **No-Go** until license, language/annotation policy, duplicate treatment, privacy review, task definition, and split strategy are approved.

No dataset is authorized for Phase 2 modeling by this document.
