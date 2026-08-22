"""Seller-handoff SLA breach risk scoring -- RESEARCH/OFFLINE scoring only.

Mirrors the hash-verified-load pattern of `FulfillmentRiskService`. See
`SellerSlaRiskRequest`/`SellerSlaRiskResponse` in schemas.py for why this
takes explicit feature input instead of an order_ref: Medusa/HASEBHA has no
seller/vendor module, so 10 of the frozen model's 22 training features
(seller-history) cannot be legitimately derived online -- see
reports/generated/olist_v3_multistage/SELLER_SLA_ONLINE_FEATURE_PARITY_AUDIT.json
for the full audit and B5 FAIL verdict this design follows.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm as lgb

from src.ai_service.config import (
    SELLER_SLA_CALIBRATOR_PATH,
    SELLER_SLA_HIGH_THRESHOLD,
    SELLER_SLA_MEDIUM_THRESHOLD,
    SELLER_SLA_MODEL_PATH,
    SELLER_SLA_MODEL_SHA256,
    SELLER_SLA_MODEL_VERSION,
)

# Exact order the frozen LightGBM booster expects (must match
# scripts/olist_v3_seller_sla_pipeline.py feature_cols exactly).
FEATURE_ORDER = [
    "days_to_shipping_deadline", "purchase_weekday", "purchase_hour", "purchase_month",
    "same_state", "n_items", "n_distinct_products", "n_categories", "total_price", "total_freight",
    "weight_g", "volume_cm3", "payment_value", "n_installments",
    "seller_past_order_count", "seller_past_breach_rate_expanding",
    "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
    "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_handling_mean_30d",
    "seller_recent_load_7d", "total_freight_over_price",
]

COLD_START_SENTINEL = -1.0


class ModelIntegrityError(RuntimeError):
    """Raised when the on-disk model artifact does not match its recorded hash."""


@dataclass(frozen=True)
class SellerSlaRiskResult:
    probability_calibrated: float
    probability_raw: float
    risk_level: str
    operational_priority_score: float
    cold_start: bool
    calibration_method: str
    model_version: str
    model_artifact_sha256: str
    features_as_of_timestamp: str
    scored_at: str
    reason_codes: list[str]


class SellerSlaRiskService:
    def __init__(self) -> None:
        if not SELLER_SLA_MODEL_PATH.exists():
            raise FileNotFoundError(f"Seller-SLA model artifact not found: {SELLER_SLA_MODEL_PATH}")
        actual_hash = hashlib.sha256(SELLER_SLA_MODEL_PATH.read_bytes()).hexdigest()
        if actual_hash != SELLER_SLA_MODEL_SHA256:
            raise ModelIntegrityError(
                f"Seller-SLA model hash mismatch: expected {SELLER_SLA_MODEL_SHA256}, got {actual_hash}"
            )
        self._model = lgb.Booster(model_file=str(SELLER_SLA_MODEL_PATH))
        self._model_artifact_sha256 = actual_hash

        self._calibration_method = "RAW"
        self._calibrator = None
        if SELLER_SLA_CALIBRATOR_PATH.exists():
            with open(SELLER_SLA_CALIBRATOR_PATH, "rb") as f:
                cal = pickle.load(f)
            self._calibration_method = cal["method"]
            self._calibrator = cal["model"]

    def _calibrate(self, p_raw: float) -> float:
        if self._calibration_method == "ISOTONIC" and self._calibrator is not None:
            return float(self._calibrator.predict([p_raw])[0])
        if self._calibration_method == "PLATT" and self._calibrator is not None:
            return float(self._calibrator.predict_proba([[p_raw]])[0][1])
        return p_raw

    def score(self, req) -> SellerSlaRiskResult:
        cold_start = not req.seller_history_available
        if cold_start:
            seller_features = {
                "seller_past_order_count": 0,
                "seller_past_breach_rate_expanding": COLD_START_SENTINEL,
                "seller_past_handling_median_expanding": COLD_START_SENTINEL,
                "seller_past_handling_std_expanding": COLD_START_SENTINEL,
                "seller_breach_rate_30d": COLD_START_SENTINEL,
                "seller_breach_rate_90d": COLD_START_SENTINEL,
                "seller_handling_mean_30d": COLD_START_SENTINEL,
                "seller_recent_load_7d": 0.0,
            }
        else:
            required = [
                "seller_past_order_count", "seller_past_breach_rate_expanding",
                "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
                "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_handling_mean_30d",
                "seller_recent_load_7d",
            ]
            missing = [f for f in required if getattr(req, f) is None]
            if missing:
                raise ValueError(
                    f"seller_history_available=True but missing required seller-history fields: {missing}"
                )
            seller_features = {f: getattr(req, f) for f in required}

        purchase_ts = req.purchase_timestamp
        row = {
            "days_to_shipping_deadline": req.days_to_shipping_deadline,
            "purchase_weekday": purchase_ts.weekday(),
            "purchase_hour": purchase_ts.hour,
            "purchase_month": purchase_ts.month,
            "same_state": int(req.same_state),
            "n_items": req.n_items,
            "n_distinct_products": req.n_distinct_products,
            "n_categories": req.n_categories,
            "total_price": req.total_price,
            "total_freight": req.total_freight,
            "weight_g": req.weight_g,
            "volume_cm3": req.volume_cm3,
            "payment_value": req.payment_value,
            "n_installments": req.n_installments,
            **seller_features,
        }
        row["total_freight_over_price"] = (
            row["total_freight"] / row["total_price"] if row["total_price"] > 0 else 0.0
        )

        ordered = [[row[name] for name in FEATURE_ORDER]]
        p_raw = float(self._model.predict(ordered)[0])
        p_cal = self._calibrate(p_raw)

        if p_cal >= SELLER_SLA_HIGH_THRESHOLD:
            risk_level = "HIGH"
        elif p_cal >= SELLER_SLA_MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        reason_codes = []
        if cold_start:
            reason_codes.append("COLD_START_NO_SELLER_HISTORY")
        if req.days_to_shipping_deadline < 2:
            reason_codes.append("TIGHT_SHIPPING_DEADLINE")
        if not cold_start and (seller_features["seller_past_breach_rate_expanding"] or 0) > 0.15:
            reason_codes.append("SELLER_ELEVATED_HISTORICAL_BREACH_RATE")

        return SellerSlaRiskResult(
            probability_calibrated=p_cal,
            probability_raw=p_raw,
            risk_level=risk_level,
            operational_priority_score=p_cal,
            cold_start=cold_start,
            calibration_method=self._calibration_method,
            model_version=SELLER_SLA_MODEL_VERSION,
            model_artifact_sha256=self._model_artifact_sha256,
            features_as_of_timestamp=purchase_ts.isoformat(),
            scored_at=datetime.now(timezone.utc).isoformat(),
            reason_codes=reason_codes,
        )
