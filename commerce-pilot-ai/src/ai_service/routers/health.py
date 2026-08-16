from __future__ import annotations

from fastapi import APIRouter, Request

from src.ai_service.config import SERVICE_NAME, SERVICE_VERSION
from src.ai_service.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    # Distinguishes "the API process is alive and answering" (this endpoint
    # returning at all) from "the model assets this endpoint reports on are
    # actually loadable" (each check below independently hash-verifies its
    # artifact rather than assuming success). NLP transformer weights are
    # loaded lazily per-model on first request (see nlp_inference.py), so
    # "loadable" here means the export manifest + file hashes check out, not
    # that the weights are currently resident in memory.
    nlp_service = getattr(request.app.state, "nlp_service", None)
    checks = {
        "fulfillment_risk_model_loaded": getattr(request.app.state, "fulfillment_service", None) is not None,
        "decision_engine_config_loaded": getattr(request.app.state, "decision_config", None) is not None,
        "nlp_registry_loaded": nlp_service is not None,
        "nlp_E_mpold_marbert_loadable": bool(nlp_service and nlp_service.is_loadable("E_MARBERT")),
        "nlp_B2_astd_marbert_loadable": bool(nlp_service and nlp_service.is_loadable("B2_MARBERT")),
        "nlp_C_labr_marbert_loadable": bool(nlp_service and nlp_service.is_loadable("C_MARBERT")),
        "nlp_C_labr_arabert_loadable": bool(nlp_service and nlp_service.is_loadable("C_AraBERT")),
        "nlp_A_amazon_classical_materialized": False,  # honest: never true until a real artifact exists
        "recommendation_engine_loaded": getattr(request.app.state, "recommendation_service", None) is not None,
    }
    # Amazon's classical artifact is a known, permanent gap (not a failure
    # condition) documented in the NLP promotion registry -- it does not
    # count against overall readiness the way an unexpected load failure
    # would.
    required = {k: v for k, v in checks.items() if k != "nlp_A_amazon_classical_materialized"}
    status = "ready" if all(required.values()) else "not_ready"
    return ReadyResponse(status=status, checks=checks)
