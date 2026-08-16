"""Pydantic request/response schemas for the CommercePilot AI service (v1)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


class FulfillmentRiskRequest(BaseModel):
    order_ref: str
    purchase_timestamp: datetime
    approval_timestamp: datetime


class FulfillmentRiskResponse(BaseModel):
    order_ref: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_class: Literal["low", "high"]
    risk_threshold: float
    risk_threshold_source: str
    model_capability: Literal["fulfillment_risk"] = "fulfillment_risk"
    model_version: str
    model_experiment_id: str
    model_artifact_sha256: str
    input_features: dict[str, float]
    scored_at: str


class DecisionRequest(BaseModel):
    customer_ref: str
    order_ref: str | None = None
    fulfillment_risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment: Literal["positive", "neutral", "negative"] | None = None
    complaint_issue_type: str | None = None
    complaint_resolved: bool | None = None
    recommendation_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    model_versions: dict[str, str] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    action: str
    priority: str
    reason_codes: list[str]
    input_signals: dict[str, object]
    model_versions: dict[str, str]
    ruleset_version: str
    created_at: str


class NotImplementedCapabilityResponse(BaseModel):
    capability: str
    status: Literal["NOT_YET_AVAILABLE"]
    reason: str


# --- NLP analyze ------------------------------------------------------------

NlpTask = Literal["E", "B2", "C", "A"]


class NlpAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    task: NlpTask
    model: Literal["marbert", "arabert"] | None = None


class NlpModelPrediction(BaseModel):
    model_key: str
    model_short_name: str
    model_name: str
    model_revision: str
    predicted_label: str
    class_probabilities: dict[str, float]
    probability_kind: Literal["model_native_softmax"] = "model_native_softmax"


class NlpAnalyzeResponse(BaseModel):
    task: str
    task_name: str
    status: Literal["OK"] = "OK"
    champion_selected: bool
    predictions: list[NlpModelPrediction]
    preprocessing_version: str
    normalized_text: str
    analyzed_at: str


class NlpArtifactNotMaterializedResponse(BaseModel):
    task: str
    task_name: str
    status: Literal["ARTIFACT_NOT_MATERIALIZED"] = "ARTIFACT_NOT_MATERIALIZED"
    frozen_config_reference: str
    reason: str


# --- Recommendations ---------------------------------------------------------


class RecommendationHistoryItem(BaseModel):
    item_id: str
    times_purchased: int = Field(ge=1)
    orders_since_last_purchase: int = Field(ge=0)


class RecommendationRequest(BaseModel):
    customer_ref: str
    history: list[RecommendationHistoryItem] = Field(default_factory=list)
    k: int = 10


class RecommendedItemResponse(BaseModel):
    item_id: str
    rank: int
    score: float
    component: Literal["reorder", "discovery_backfill"]
    reason_code: str
    cross_catalog_deployment_status: Literal["VALID_ANY_CATALOG", "NOT_YET_VALIDATED"]


class RecommendationResponse(BaseModel):
    customer_ref: str
    items: list[RecommendedItemResponse]
    k: int
    recommender_version: str
    ranking_logic_sha256: str
    popularity_artifact_sha256: str
    catalog_domain: Literal["instacart_offline_validated"]
    egyptian_catalog_mapping_status: Literal["NOT_YET_VALIDATED"] = "NOT_YET_VALIDATED"
    generated_at: str
