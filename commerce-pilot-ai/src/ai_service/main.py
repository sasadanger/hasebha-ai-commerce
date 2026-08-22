"""CommercePilot AI service — FastAPI entrypoint.

Run from the repo root (module imports are `src.ai_service.*`, matching the
rest of the codebase's test/import convention):
    uvicorn src.ai_service.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.ai_service.config import SERVICE_NAME, SERVICE_VERSION
from src.ai_service.routers import decision, fulfillment, health, nlp, recommendations
from src.ai_service.services.decision_engine import DecisionEngineConfig
from src.ai_service.services.fulfillment_risk import FulfillmentRiskService
from src.ai_service.services.nlp_inference import NlpInferenceService
from src.ai_service.services.recommendation_engine import RecommendationEngineService
from src.ai_service.services.seller_sla_risk import SellerSlaRiskService
from src.ai_service.services.production_parity_seller_sla import ProductionParitySellerSlaService
from src.ai_service.services.prediction_feedback_store import PredictionFeedbackStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup; failures degrade the relevant /ready check to
    # false rather than crashing the whole service, so an artifact/config
    # problem in one capability cannot take down the others.
    #
    # Only the CatBoost fulfillment model, the decision config, and the NLP/
    # recommendation *registries* (manifest read + file hash checks, no
    # transformer weights) load eagerly here -- transformer weights load
    # lazily on first /v1/nlp/analyze request per model, per the resource
    # discipline in reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/
    # and the broader "one heavy workload at a time" policy for this project.
    try:
        app.state.fulfillment_service = FulfillmentRiskService()
    except Exception:
        logger.exception("failed to load fulfillment risk model")
        app.state.fulfillment_service = None

    try:
        app.state.seller_sla_service = SellerSlaRiskService()
    except Exception:
        logger.exception("failed to load seller-SLA risk model")
        app.state.seller_sla_service = None

    try:
        app.state.production_parity_seller_sla_service = ProductionParitySellerSlaService()
    except Exception:
        logger.exception("failed to load production-parity seller-SLA shadow model")
        app.state.production_parity_seller_sla_service = None

    try:
        app.state.seller_sla_feedback_store = PredictionFeedbackStore()
    except Exception:
        logger.exception("failed to init seller-SLA feedback store")
        app.state.seller_sla_feedback_store = None

    try:
        app.state.decision_config = DecisionEngineConfig.load()
    except Exception:
        logger.exception("failed to load decision engine config")
        app.state.decision_config = None

    try:
        app.state.nlp_service = NlpInferenceService()
    except Exception:
        logger.exception("failed to load NLP inference registry")
        app.state.nlp_service = None

    try:
        app.state.recommendation_service = RecommendationEngineService()
    except Exception:
        logger.exception("failed to load recommendation engine")
        app.state.recommendation_service = None

    yield


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)

app.include_router(health.router)
app.include_router(fulfillment.router)
app.include_router(decision.router)
app.include_router(nlp.router)
app.include_router(recommendations.router)
