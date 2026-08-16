"""Transparent, configuration-driven Business Decision Engine (Phase 4/Z).

Deliberately not a trained model: the brief requires reason-coded,
explainable decisions, so this is an ordered rule evaluator over declared
thresholds loaded from `configs/decision_engine_rules.yaml`.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.ai_service.config import DECISION_ENGINE_CONFIG_PATH


@dataclass(frozen=True)
class DecisionEngineConfig:
    fulfillment_risk_high: float
    recommendation_strength_strong: float
    service_priority_complaint_types: frozenset[str]
    ruleset_version: str

    @staticmethod
    def load(path: Path = DECISION_ENGINE_CONFIG_PATH) -> "DecisionEngineConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        thresholds = data["thresholds"]
        return DecisionEngineConfig(
            fulfillment_risk_high=float(thresholds["fulfillment_risk_high"]),
            recommendation_strength_strong=float(thresholds["recommendation_strength_strong"]),
            service_priority_complaint_types=frozenset(data["service_priority_complaint_types"]),
            ruleset_version=str(data["schema_version"]),
        )


@dataclass(frozen=True)
class DecisionInput:
    customer_ref: str
    order_ref: str | None = None
    fulfillment_risk_score: float | None = None
    sentiment: str | None = None  # "positive" | "neutral" | "negative"
    complaint_issue_type: str | None = None
    complaint_resolved: bool | None = None
    recommendation_strength: float | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    action: str
    priority: str
    reason_codes: Sequence[str]
    input_signals: Mapping[str, object]
    model_versions: Mapping[str, str]
    ruleset_version: str
    created_at: str


def evaluate(decision_input: DecisionInput, config: DecisionEngineConfig) -> DecisionResult:
    risk = decision_input.fulfillment_risk_score
    high_risk = risk is not None and risk >= config.fulfillment_risk_high
    is_service_complaint = (
        decision_input.complaint_issue_type is not None
        and decision_input.complaint_issue_type in config.service_priority_complaint_types
    )
    negative_unresolved = (
        decision_input.sentiment == "negative" and decision_input.complaint_resolved is False
    )
    strong_recommendation = (
        decision_input.recommendation_strength is not None
        and decision_input.recommendation_strength >= config.recommendation_strength_strong
    )

    # Ordered by severity: a service-affecting complaint compounded by high
    # fulfillment risk always wins, ahead of risk alone, ahead of unresolved
    # negative sentiment suppressing marketing, ahead of a positive cross-sell
    # signal. This ordering is the explicit answer to the brief's required
    # conflict test: high risk + delivery complaint outranks a merely "good"
    # customer state with a strong recommendation.
    if high_risk and is_service_complaint:
        action, priority = "ESCALATE_TO_HUMAN_SERVICE", "P0_CRITICAL"
        reason_codes = ["HIGH_FULFILLMENT_RISK", "ACTIVE_SERVICE_AFFECTING_COMPLAINT"]
    elif high_risk:
        action, priority = "INTERVENTION_PRIORITY", "P1_HIGH"
        reason_codes = ["HIGH_FULFILLMENT_RISK"]
    elif negative_unresolved:
        action, priority = "SUPPRESS_CROSS_SELL", "P2_SUPPRESS_MARKETING"
        reason_codes = ["NEGATIVE_UNRESOLVED_COMPLAINT"]
    elif strong_recommendation:
        action, priority = "ALLOW_CROSS_SELL", "P3_GROWTH"
        reason_codes = ["STRONG_RECOMMENDATION", "ACCEPTABLE_FULFILLMENT_RISK"]
    else:
        action, priority = "NO_ACTION", "P4_ROUTINE"
        reason_codes = ["NO_TRIGGERING_SIGNAL"]

    return DecisionResult(
        action=action,
        priority=priority,
        reason_codes=reason_codes,
        input_signals={
            "customer_ref": decision_input.customer_ref,
            "order_ref": decision_input.order_ref,
            "fulfillment_risk_score": risk,
            "sentiment": decision_input.sentiment,
            "complaint_issue_type": decision_input.complaint_issue_type,
            "complaint_resolved": decision_input.complaint_resolved,
            "recommendation_strength": decision_input.recommendation_strength,
        },
        model_versions=dict(decision_input.model_versions),
        ruleset_version=config.ruleset_version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
