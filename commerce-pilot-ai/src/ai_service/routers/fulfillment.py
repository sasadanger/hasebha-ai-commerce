from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.ai_service.auth import require_api_key
from src.ai_service.schemas import FulfillmentRiskRequest, FulfillmentRiskResponse

router = APIRouter(prefix="/v1/fulfillment", tags=["fulfillment"], dependencies=[Depends(require_api_key)])


@router.post("/risk", response_model=FulfillmentRiskResponse)
def score_fulfillment_risk(payload: FulfillmentRiskRequest, request: Request) -> FulfillmentRiskResponse:
    service = getattr(request.app.state, "fulfillment_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="fulfillment risk model unavailable")
    try:
        result = service.score(
            purchase_timestamp=payload.purchase_timestamp,
            approval_timestamp=payload.approval_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FulfillmentRiskResponse(
        order_ref=payload.order_ref,
        risk_score=result.risk_score,
        risk_class=result.risk_class,
        risk_threshold=result.risk_threshold,
        risk_threshold_source=result.risk_threshold_source,
        model_version=result.model_version,
        model_experiment_id=result.model_experiment_id,
        model_artifact_sha256=result.model_artifact_sha256,
        input_features=result.input_features,
        scored_at=result.scored_at,
    )
