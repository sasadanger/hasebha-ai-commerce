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


# --- Seller SLA risk (research/offline scoring only -- see
# reports/generated/olist_v3_multistage/SELLER_SLA_ONLINE_FEATURE_PARITY_AUDIT.json.
# Medusa/HASEBHA is a single-vendor store with no seller/vendor module, so 10 of
# the model's 22 training features (all seller-history features) cannot be
# legitimately derived from a live Medusa order. This endpoint therefore takes
# explicit feature input rather than an order_ref -- it is NOT wired to Medusa's
# order.placed event, and callers must supply real historical values or the
# request is rejected. It exists to (a) allow manual/offline scoring against the
# frozen model, and (b) establish the response contract + persistence/feedback
# plumbing ahead of a future first-party retraining once real seller-history
# data exists. ------------------------------------------------------------


class SellerSlaRiskRequest(BaseModel):
    order_ref: str | None = None
    seller_ref: str | None = None
    purchase_timestamp: datetime
    seller_history_available: bool
    # Order/product features -- always required, always legitimately derivable
    # online from a real Medusa order (see parity audit: PARITY_STATUS EXACT or
    # DERIVABLE_EQUIVALENT for all of these).
    days_to_shipping_deadline: float = Field(ge=0.0, le=365.0)
    n_items: int = Field(ge=1, le=1000)
    n_distinct_products: int = Field(ge=1, le=1000)
    n_categories: int = Field(ge=1, le=1000)
    total_price: float = Field(ge=0.0)
    total_freight: float = Field(ge=0.0)
    weight_g: float = Field(ge=0.0)
    volume_cm3: float = Field(ge=0.0)
    payment_value: float = Field(ge=0.0)
    n_installments: int = Field(ge=1, le=60)
    same_state: bool = False
    # Seller-history features -- REQUIRED if seller_history_available=True (a
    # real seller with real prior orders); if False, the service applies the
    # documented cold-start sentinel (-1.0) itself, exactly matching the
    # training-time cold-start convention, rather than accepting caller-supplied
    # fake defaults for a seller with no real history.
    seller_past_order_count: int | None = Field(default=None, ge=0)
    seller_past_breach_rate_expanding: float | None = Field(default=None, ge=0.0, le=1.0)
    seller_past_handling_median_expanding: float | None = None
    seller_past_handling_std_expanding: float | None = Field(default=None, ge=0.0)
    seller_breach_rate_30d: float | None = Field(default=None, ge=0.0, le=1.0)
    seller_breach_rate_90d: float | None = Field(default=None, ge=0.0, le=1.0)
    seller_handling_mean_30d: float | None = None
    seller_recent_load_7d: float | None = Field(default=None, ge=0.0)


class SellerSlaRiskResponse(BaseModel):
    order_ref: str | None = None
    prediction_stage: Literal["T0"] = "T0"
    seller_sla_breach_probability: float = Field(ge=0.0, le=1.0)
    seller_sla_breach_probability_raw: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    operational_priority_score: float = Field(ge=0.0, le=1.0)
    seller_history_available: bool
    cold_start: bool
    calibration_method: str
    feature_parity_status: Literal["RESEARCH_OFFLINE_ONLY"] = "RESEARCH_OFFLINE_ONLY"
    model_name: str
    model_version: str
    model_artifact_sha256: str
    features_as_of_timestamp: str
    scored_at: str
    reason_codes: list[str] = Field(default_factory=list)


# --- Production-parity Seller-SLA SHADOW MODE (Gates 3/6/9/10 -- a DIFFERENT
# model from the RESEARCH SellerSlaRiskRequest/Response above. That research
# endpoint scores the frozen 22-feature model against manually-supplied
# explicit features (never live Medusa data). THIS endpoint scores the new
# production-parity model (13 features, all genuinely available online in a
# single-vendor Medusa/HASEBHA store -- see
# reports/generated/olist_v3_multistage/SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json)
# and is the one intended to be called from a real order.placed event, in
# SHADOW MODE ONLY: predictions are persisted for observation, never used to
# block checkout, cancel orders, or trigger customer-facing changes. The two
# routes/schemas are named and typed distinctly on purpose so neither an
# admin nor a future engineer can mistake one for the other. -----------------


class ProductionParitySellerSlaShadowRequest(BaseModel):
    order_ref: str
    purchase_timestamp: datetime
    n_items: int = Field(ge=1, le=1000)
    n_distinct_products: int = Field(ge=1, le=1000)
    n_categories: int = Field(ge=1, le=1000)
    total_price: float = Field(ge=0.0)
    total_freight: float = Field(ge=0.0)
    weight_g: float = Field(ge=0.0)
    volume_cm3: float = Field(ge=0.0)
    payment_value: float = Field(ge=0.0)
    same_zone: bool = Field(
        description=(
            "Whether the order ships from the same zone/province as the customer's shipping "
            "address, per the fulfilling StockLocation -- the honest single-vendor analog of "
            "Olist's seller/customer same_state feature, NOT claimed identical to it."
        )
    )


class ProductionParitySellerSlaShadowResponse(BaseModel):
    order_ref: str
    prediction_stage: Literal["T0"] = "T0"
    prediction_mode: Literal["SHADOW"] = "SHADOW"
    seller_sla_breach_probability: float = Field(ge=0.0, le=1.0)
    seller_sla_breach_probability_raw: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    calibration_method: str
    model_name: Literal["olist_production_parity_seller_sla"] = "olist_production_parity_seller_sla"
    model_version: str
    model_artifact_sha256: str
    features_as_of_timestamp: str
    scored_at: str
    automated_action_taken: Literal[False] = False
    persisted: bool


# --- NLP analyze ------------------------------------------------------------

NlpTask = Literal["E", "B2", "C", "A"]


class NlpAnalyzeRequest(BaseModel):
    # max_length bounds request size before it reaches tokenization/model
    # inference -- without it, an oversized payload is a resource-exhaustion
    # vector. 10,000 chars comfortably covers real review/comment text.
    text: str = Field(min_length=1, max_length=10_000)
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
    # Bounded for the same reason as NlpAnalyzeRequest.text: an unbounded
    # list/k is a resource-exhaustion vector against the ranking engine.
    # 1000 items is far beyond any real customer's purchase history; 100
    # results is far beyond any real storefront recommendation slot.
    history: list[RecommendationHistoryItem] = Field(default_factory=list, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)


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
