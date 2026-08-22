"""Production-parity Seller-SLA model -- SHADOW MODE ONLY.

A DIFFERENT model from `SellerSlaRiskService` (the 22-feature RESEARCH_OFFLINE_ONLY
model). This one is trained only on the 13 features Gate 1's re-audit found
genuinely available online in a single-vendor Medusa/HASEBHA store. Its
temporal signal is WEAK (mean AUC ~0.555, worst ~0.529 -- see
PRODUCTION_PARITY_MODEL_COMPARISON.json), so it is wired for shadow-mode
prediction logging only, never for any automated decision, per the mission's
explicit Gate 9 rule.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm as lgb

from src.ai_service.config import (
    PRODUCTION_PARITY_CALIBRATOR_PATH,
    PRODUCTION_PARITY_HIGH_THRESHOLD,
    PRODUCTION_PARITY_MEDIUM_THRESHOLD,
    PRODUCTION_PARITY_MODEL_PATH,
    PRODUCTION_PARITY_MODEL_SHA256,
    PRODUCTION_PARITY_MODEL_VERSION,
)

# Exact order the frozen production-parity booster expects.
FEATURE_ORDER = [
    "purchase_weekday", "purchase_hour", "purchase_month", "same_state",
    "n_items", "n_distinct_products", "n_categories", "total_price", "total_freight",
    "total_freight_over_price", "weight_g", "volume_cm3", "payment_value",
]


class ModelIntegrityError(RuntimeError):
    """Raised when the on-disk model artifact does not match its recorded hash."""


@dataclass(frozen=True)
class ProductionParityResult:
    probability_calibrated: float
    probability_raw: float
    risk_level: str
    calibration_method: str
    model_version: str
    model_artifact_sha256: str
    features_as_of_timestamp: str
    scored_at: str


class ProductionParitySellerSlaService:
    def __init__(self) -> None:
        if not PRODUCTION_PARITY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Production-parity model artifact not found: {PRODUCTION_PARITY_MODEL_PATH}"
            )
        actual_hash = hashlib.sha256(PRODUCTION_PARITY_MODEL_PATH.read_bytes()).hexdigest()
        if actual_hash != PRODUCTION_PARITY_MODEL_SHA256:
            raise ModelIntegrityError(
                f"Production-parity model hash mismatch: expected {PRODUCTION_PARITY_MODEL_SHA256}, "
                f"got {actual_hash}"
            )
        self._model = lgb.Booster(model_file=str(PRODUCTION_PARITY_MODEL_PATH))
        self._model_artifact_sha256 = actual_hash

        self._calibration_method = "RAW"
        self._calibrator = None
        if PRODUCTION_PARITY_CALIBRATOR_PATH.exists():
            with open(PRODUCTION_PARITY_CALIBRATOR_PATH, "rb") as f:
                cal = pickle.load(f)
            self._calibration_method = cal["method"]
            self._calibrator = cal["model"]

    def _calibrate(self, p_raw: float) -> float:
        if self._calibration_method == "ISOTONIC" and self._calibrator is not None:
            return float(self._calibrator.predict([p_raw])[0])
        if self._calibration_method == "PLATT" and self._calibrator is not None:
            return float(self._calibrator.predict_proba([[p_raw]])[0][1])
        return p_raw

    def score(self, req) -> ProductionParityResult:
        purchase_ts = req.purchase_timestamp
        total_price = req.total_price
        total_freight_over_price = (req.total_freight / total_price) if total_price > 0 else 0.0
        row = {
            "purchase_weekday": purchase_ts.weekday(),
            "purchase_hour": purchase_ts.hour,
            "purchase_month": purchase_ts.month,
            "same_state": int(req.same_zone),
            "n_items": req.n_items,
            "n_distinct_products": req.n_distinct_products,
            "n_categories": req.n_categories,
            "total_price": total_price,
            "total_freight": req.total_freight,
            "total_freight_over_price": total_freight_over_price,
            "weight_g": req.weight_g,
            "volume_cm3": req.volume_cm3,
            "payment_value": req.payment_value,
        }
        ordered = [[row[name] for name in FEATURE_ORDER]]
        p_raw = float(self._model.predict(ordered)[0])
        p_cal = self._calibrate(p_raw)

        if p_cal >= PRODUCTION_PARITY_HIGH_THRESHOLD:
            risk_level = "HIGH"
        elif p_cal >= PRODUCTION_PARITY_MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return ProductionParityResult(
            probability_calibrated=p_cal,
            probability_raw=p_raw,
            risk_level=risk_level,
            calibration_method=self._calibration_method,
            model_version=PRODUCTION_PARITY_MODEL_VERSION,
            model_artifact_sha256=self._model_artifact_sha256,
            features_as_of_timestamp=purchase_ts.isoformat(),
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
