"""Minimal append-only prediction/outcome feedback store (Part E).

Design rationale: `ai_service` is currently a stateless FastAPI process with
no database of its own (confirmed by inspection -- no sqlite/postgres/
sqlalchemy anywhere in src/ai_service/). The existing, PRODUCTION persistence
pattern for AI predictions is the Medusa `order.metadata` write in
`order-placed.ts` (see that file's docstring), which is the right place for
any model that is actually wired to a live Medusa event.

The seller-SLA model is NOT wired to a live Medusa event this session (see
SELLER_SLA_ONLINE_FEATURE_PARITY_AUDIT.json, PARITY=FAIL) -- so there is no
Medusa order to attach metadata to automatically. Per the mission's own Part
E guidance ("implement the smallest append-only prediction record required
... do not overbuild event-stream infrastructure"), this module is a
deliberately small JSON-Lines append-only local store, NOT a new database
table/migration, so that:
  (a) research/offline scoring calls made through the new route are still
      captured as durable evidence (rather than being lost after the
      response is returned), and
  (b) once a live Medusa event legitimately supplies real seller-SLA
      features (i.e. once B5's FAIL verdict is superseded by real
      first-party seller-history data), the same record_prediction /
      record_outcome contract can be reused, or ported to a real DB table,
      without redesigning the call sites.

This is explicitly NOT the persistence layer for the existing, live V1
`/v1/fulfillment/risk` -> order.placed integration, which continues to use
its own proven `order.metadata` pattern unchanged.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

DEFAULT_STORE_PATH = Path("artifacts/experiments/olist_v3_multistage/seller_sla_predictions.jsonl")

_LOCK = Lock()


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    order_id: str | None
    seller_id: str | None
    prediction_type: str
    prediction_stage: str
    prediction_timestamp: str
    features_as_of: str
    score: float
    risk_level: str
    model_name: str
    model_version: str
    artifact_sha256: str
    cold_start: bool
    decision: str | None = None
    outcome: dict | None = None
    # Added for the "raw feature/prediction persistence" gap identified in
    # HASEBHA_FULFILLMENT_FEEDBACK_DATASET_CONTRACT.md: without this, a
    # future retrain could see the SCORE a model produced but not the exact
    # inputs it produced that score from, making the record useless as a
    # training example. Optional/defaulted so existing rows (written before
    # this field existed) and any caller not yet passing it remain valid --
    # purely additive, does not change the meaning of any existing field.
    raw_features: dict | None = None
    feature_schema_version: str | None = None


class PredictionFeedbackStore:
    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        *,
        order_id: str | None,
        seller_id: str | None,
        prediction_stage: str,
        features_as_of: str,
        score: float,
        risk_level: str,
        model_name: str,
        model_version: str,
        artifact_sha256: str,
        cold_start: bool,
        decision: str | None = None,
        raw_features: dict | None = None,
        feature_schema_version: str | None = None,
    ) -> PredictionRecord:
        record = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            order_id=order_id,
            seller_id=seller_id,
            prediction_type="seller_sla_breach",
            prediction_stage=prediction_stage,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            features_as_of=features_as_of,
            score=score,
            risk_level=risk_level,
            model_name=model_name,
            model_version=model_version,
            artifact_sha256=artifact_sha256,
            cold_start=cold_start,
            decision=decision,
            raw_features=raw_features,
            feature_schema_version=feature_schema_version,
        )
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        return record

    def record_outcome(
        self,
        prediction_id: str,
        *,
        seller_sla_breached_actual: bool | None = None,
        carrier_handoff_time_actual: str | None = None,
        customer_late_actual: bool | None = None,
    ) -> bool:
        """Append an outcome-linked record. Append-only store: rather than
        mutating the original line (which would require a real DB), this
        writes a new record referencing the same prediction_id with an
        `outcome` payload -- a reconciliation job (not implemented this
        session, out of scope per Part E3 'no automatic online learning')
        would join on prediction_id when building a training set later."""
        if not self.path.exists():
            return False
        outcome_observed_at = datetime.now(timezone.utc).isoformat()
        outcome_line = {
            "prediction_id": prediction_id,
            "outcome": {
                "seller_sla_breached_actual": seller_sla_breached_actual,
                "carrier_handoff_time_actual": carrier_handoff_time_actual,
                "customer_late_actual": customer_late_actual,
                "outcome_observed_at": outcome_observed_at,
            },
        }
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome_line) + "\n")
        return True

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
