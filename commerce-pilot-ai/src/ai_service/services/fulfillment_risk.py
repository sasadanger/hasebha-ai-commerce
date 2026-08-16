"""Fulfillment-risk scoring backed by the frozen Olist Phase 2A CatBoost champion.

Loads the model once (hash-verified against the value recorded in
`selection_manifest.json` at load time) and exposes a pure scoring function
over the 9 strict-core, as-of-safe timing features the model was trained on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from catboost import CatBoostClassifier

from src.ai_service.config import (
    OLIST_EXPERIMENT_ID,
    OLIST_MODEL_PATH,
    OLIST_MODEL_SHA256,
    OLIST_RISK_THRESHOLD,
    OLIST_RISK_THRESHOLD_SOURCE,
)

STRICT_FEATURES = (
    "purchase_year",
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "approval_year",
    "approval_month",
    "approval_day_of_week",
    "approval_hour",
    "purchase_to_approval_seconds",
)


class ModelIntegrityError(RuntimeError):
    """Raised when the on-disk model artifact does not match its recorded hash."""


@dataclass(frozen=True)
class FulfillmentRiskResult:
    risk_score: float
    risk_class: str
    risk_threshold: float
    risk_threshold_source: str
    model_version: str
    model_experiment_id: str
    model_artifact_sha256: str
    input_features: dict
    scored_at: str


class FulfillmentRiskService:
    def __init__(self) -> None:
        if not OLIST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Olist model artifact not found: {OLIST_MODEL_PATH}")
        actual_hash = hashlib.sha256(OLIST_MODEL_PATH.read_bytes()).hexdigest()
        if actual_hash != OLIST_MODEL_SHA256:
            raise ModelIntegrityError(
                f"Olist model hash mismatch: expected {OLIST_MODEL_SHA256}, got {actual_hash}"
            )
        self._model = CatBoostClassifier()
        self._model.load_model(str(OLIST_MODEL_PATH))
        self._model_version = actual_hash[:12]
        self._model_artifact_sha256 = actual_hash

    def score(
        self,
        *,
        purchase_timestamp: datetime,
        approval_timestamp: datetime,
    ) -> FulfillmentRiskResult:
        if approval_timestamp < purchase_timestamp:
            raise ValueError("approval_timestamp cannot precede purchase_timestamp")
        purchase_to_approval_seconds = (approval_timestamp - purchase_timestamp).total_seconds()
        features = {
            "purchase_year": purchase_timestamp.year,
            "purchase_month": purchase_timestamp.month,
            "purchase_day_of_week": purchase_timestamp.isoweekday() % 7,
            "purchase_hour": purchase_timestamp.hour,
            "approval_year": approval_timestamp.year,
            "approval_month": approval_timestamp.month,
            "approval_day_of_week": approval_timestamp.isoweekday() % 7,
            "approval_hour": approval_timestamp.hour,
            "purchase_to_approval_seconds": purchase_to_approval_seconds,
        }
        row = [[features[name] for name in STRICT_FEATURES]]
        risk_score = float(self._model.predict_proba(row)[0][1])
        risk_class = "high" if risk_score >= OLIST_RISK_THRESHOLD else "low"
        return FulfillmentRiskResult(
            risk_score=risk_score,
            risk_class=risk_class,
            risk_threshold=OLIST_RISK_THRESHOLD,
            risk_threshold_source=OLIST_RISK_THRESHOLD_SOURCE,
            model_version=self._model_version,
            model_experiment_id=OLIST_EXPERIMENT_ID,
            model_artifact_sha256=self._model_artifact_sha256,
            input_features=features,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
