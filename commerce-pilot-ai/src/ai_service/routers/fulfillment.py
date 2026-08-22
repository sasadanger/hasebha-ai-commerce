from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.ai_service.auth import require_api_key
from src.ai_service.schemas import (
    FulfillmentRiskRequest,
    FulfillmentRiskResponse,
    SellerSlaRiskRequest,
    SellerSlaRiskResponse,
    ProductionParitySellerSlaShadowRequest,
    ProductionParitySellerSlaShadowResponse,
)

logger = logging.getLogger(__name__)

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


@router.post("/seller-sla-risk", response_model=SellerSlaRiskResponse)
def score_seller_sla_risk(payload: SellerSlaRiskRequest, request: Request) -> SellerSlaRiskResponse:
    """RESEARCH/OFFLINE scoring endpoint for the Olist seller-SLA breach model.

    NOT wired to Medusa's order.placed event -- see
    reports/generated/olist_v3_multistage/SELLER_SLA_ONLINE_FEATURE_PARITY_AUDIT.json
    (SELLER_SLA_ONLINE_FEATURE_PARITY=FAIL: HASEBHA has no seller/vendor module,
    so seller-history features cannot be legitimately derived online). Callers
    must supply real feature values; this never fabricates defaults.
    """
    service = getattr(request.app.state, "seller_sla_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="seller-SLA risk model unavailable")
    try:
        result = service.score(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    feedback_store = getattr(request.app.state, "seller_sla_feedback_store", None)
    if feedback_store is not None:
        try:
            feedback_store.record_prediction(
                order_id=payload.order_ref,
                seller_id=payload.seller_ref,
                prediction_stage="T0",
                features_as_of=result.features_as_of_timestamp,
                score=result.probability_calibrated,
                risk_level=result.risk_level,
                model_name="olist_seller_sla",
                model_version=result.model_version,
                artifact_sha256=result.model_artifact_sha256,
                cold_start=result.cold_start,
                raw_features=payload.model_dump(mode="json"),
                feature_schema_version="seller_sla_risk_request_v1",
            )
        except Exception:
            # Persistence is best-effort (scoring must not fail because
            # logging failed), but the failure itself must NEVER be silent --
            # structured logging with enough context to investigate/replay.
            logger.warning(
                "seller_sla_risk_prediction_persist_failed",
                extra={
                    "order_ref": payload.order_ref,
                    "seller_ref": payload.seller_ref,
                    "route": "/v1/fulfillment/seller-sla-risk",
                },
                exc_info=True,
            )

    return SellerSlaRiskResponse(
        order_ref=payload.order_ref,
        seller_sla_breach_probability=result.probability_calibrated,
        seller_sla_breach_probability_raw=result.probability_raw,
        risk_level=result.risk_level,
        operational_priority_score=result.operational_priority_score,
        seller_history_available=payload.seller_history_available,
        cold_start=result.cold_start,
        calibration_method=result.calibration_method,
        model_name="olist_seller_sla",
        model_version=result.model_version,
        model_artifact_sha256=result.model_artifact_sha256,
        features_as_of_timestamp=result.features_as_of_timestamp,
        scored_at=result.scored_at,
        reason_codes=result.reason_codes,
    )


@router.post("/seller-sla-shadow", response_model=ProductionParitySellerSlaShadowResponse)
def score_seller_sla_shadow(
    payload: ProductionParitySellerSlaShadowRequest, request: Request
) -> ProductionParitySellerSlaShadowResponse:
    """PRODUCTION SHADOW-MODE scoring for the production-parity Seller-SLA model.

    Intended to be called from Medusa's `order.placed` subscriber with real
    order data. WEAK signal (mean AUC ~0.555 -- see
    PRODUCTION_PARITY_MODEL_COMPARISON.json): this endpoint exists to build
    the first-party feedback dataset (Gate 12), NOT as a validated risk
    signal. `automated_action_taken` is always False and no caller may ever
    treat this response as authorization for any automated decision. This is
    a DIFFERENT route/model from `/seller-sla-risk` (the 22-feature research
    model, explicit-feature-input only) -- do not confuse the two.
    """
    service = getattr(request.app.state, "production_parity_seller_sla_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="production-parity seller-SLA shadow model unavailable")
    try:
        result = service.score(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    persisted = False
    feedback_store = getattr(request.app.state, "seller_sla_feedback_store", None)
    if feedback_store is not None:
        try:
            feedback_store.record_prediction(
                order_id=payload.order_ref,
                seller_id=None,
                prediction_stage="T0",
                features_as_of=result.features_as_of_timestamp,
                score=result.probability_calibrated,
                risk_level=result.risk_level,
                model_name="olist_production_parity_seller_sla",
                model_version=result.model_version,
                artifact_sha256=result.model_artifact_sha256,
                cold_start=False,
                raw_features=payload.model_dump(mode="json"),
                feature_schema_version="production_parity_seller_sla_shadow_request_v1",
            )
            persisted = True
        except Exception:
            logger.warning(
                "production_parity_seller_sla_shadow_persist_failed",
                extra={"order_ref": payload.order_ref, "route": "/v1/fulfillment/seller-sla-shadow"},
                exc_info=True,
            )

    return ProductionParitySellerSlaShadowResponse(
        order_ref=payload.order_ref,
        seller_sla_breach_probability=result.probability_calibrated,
        seller_sla_breach_probability_raw=result.probability_raw,
        risk_level=result.risk_level,
        calibration_method=result.calibration_method,
        model_version=result.model_version,
        model_artifact_sha256=result.model_artifact_sha256,
        features_as_of_timestamp=result.features_as_of_timestamp,
        scored_at=result.scored_at,
        persisted=persisted,
    )
