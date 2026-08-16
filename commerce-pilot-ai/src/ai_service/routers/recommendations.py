"""Recommendation endpoint.

Real inference using the frozen Instacart candidate
`hybrid_with_popularity_backfill`. Runs regardless of whether a real-store
catalog mapping exists -- the engine executes on any correctly-shaped
history input; only the discovery/backfill slots (Instacart-catalog
popularity) are marked NOT_YET_VALIDATED for cross-catalog deployment.
Reorder slots (ranked purely from the caller's own history) are valid for
any catalog. See src/ai_service/services/recommendation_engine.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.ai_service.auth import require_api_key

from src.ai_service.schemas import RecommendationRequest, RecommendationResponse, RecommendedItemResponse

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="recommendation engine unavailable")

    try:
        result = service.recommend(
            history=[h.model_dump() for h in payload.history],
            k=payload.k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RecommendationResponse(
        customer_ref=payload.customer_ref,
        items=[
            RecommendedItemResponse(
                item_id=item.item_id,
                rank=item.rank,
                score=item.score,
                component=item.component,
                reason_code=item.reason_code,
                cross_catalog_deployment_status=item.cross_catalog_deployment_status,
            )
            for item in result.items
        ],
        k=result.k,
        recommender_version=result.recommender_version,
        ranking_logic_sha256=result.ranking_logic_sha256,
        popularity_artifact_sha256=result.popularity_artifact_sha256,
        catalog_domain=result.catalog_domain,
        generated_at=result.generated_at,
    )
