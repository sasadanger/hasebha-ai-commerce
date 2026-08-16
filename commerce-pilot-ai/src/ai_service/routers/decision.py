from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.ai_service.schemas import DecisionRequest, DecisionResponse
from src.ai_service.services.decision_engine import DecisionInput, evaluate

router = APIRouter(prefix="/v1/decision", tags=["decision"])


@router.post("", response_model=DecisionResponse)
def make_decision(payload: DecisionRequest, request: Request) -> DecisionResponse:
    config = getattr(request.app.state, "decision_config", None)
    if config is None:
        raise HTTPException(status_code=503, detail="decision engine configuration unavailable")
    result = evaluate(
        DecisionInput(
            customer_ref=payload.customer_ref,
            order_ref=payload.order_ref,
            fulfillment_risk_score=payload.fulfillment_risk_score,
            sentiment=payload.sentiment,
            complaint_issue_type=payload.complaint_issue_type,
            complaint_resolved=payload.complaint_resolved,
            recommendation_strength=payload.recommendation_strength,
            model_versions=payload.model_versions,
        ),
        config,
    )
    return DecisionResponse(
        action=result.action,
        priority=result.priority,
        reason_codes=list(result.reason_codes),
        input_signals=dict(result.input_signals),
        model_versions=dict(result.model_versions),
        ruleset_version=result.ruleset_version,
        created_at=result.created_at,
    )
