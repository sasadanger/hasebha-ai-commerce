"""Tests for ProductionParitySellerSlaService and /v1/fulfillment/seller-sla-shadow."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.ai_service.main import app
from src.ai_service.services.production_parity_seller_sla import (
    ProductionParitySellerSlaService,
    ModelIntegrityError,
    FEATURE_ORDER,
)
from tests.conftest import TEST_API_KEY


@pytest.fixture()
def client():
    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        yield c


def _payload(**overrides):
    payload = {
        "order_ref": "order_shadow_1",
        "purchase_timestamp": "2024-03-15T10:00:00",
        "n_items": 2,
        "n_distinct_products": 2,
        "n_categories": 1,
        "total_price": 150.0,
        "total_freight": 20.0,
        "weight_g": 800.0,
        "volume_cm3": 4000.0,
        "payment_value": 170.0,
        "same_zone": False,
    }
    payload.update(overrides)
    return payload


# --- feature contract ---------------------------------------------------------


def test_feature_order_excludes_forbidden_fields():
    # No seller-history fields, no shipping-deadline field, no n_installments --
    # the production-parity feature contract must never include a feature this
    # session's re-audit found NOT_AVAILABLE in HASEBHA.
    forbidden = {
        "seller_past_order_count", "seller_past_breach_rate_expanding",
        "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_recent_load_7d",
        "days_to_shipping_deadline", "n_installments",
    }
    assert forbidden.isdisjoint(set(FEATURE_ORDER))


def test_feature_order_has_exactly_13_features():
    assert len(FEATURE_ORDER) == 13


# --- service-level ---------------------------------------------------------


def test_service_loads_and_hash_matches():
    svc = ProductionParitySellerSlaService()
    assert svc is not None


def test_service_scores_valid_probability():
    svc = ProductionParitySellerSlaService()

    class Req:
        pass

    req = Req()
    payload = _payload()
    for k, v in payload.items():
        if k == "purchase_timestamp":
            v = datetime.fromisoformat(v)
        if k == "same_zone":
            setattr(req, "same_zone", v)
            continue
        setattr(req, k, v)
    result = svc.score(req)
    assert 0.0 <= result.probability_calibrated <= 1.0
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}


def test_service_rejects_tampered_artifact(tmp_path, monkeypatch):
    import src.ai_service.services.production_parity_seller_sla as mod

    fake_model = tmp_path / "fake.txt"
    fake_model.write_text("not a real model")
    monkeypatch.setattr(mod, "PRODUCTION_PARITY_MODEL_PATH", fake_model)
    with pytest.raises(ModelIntegrityError):
        mod.ProductionParitySellerSlaService()


# --- route-level -------------------------------------------------------------


def test_route_returns_200_and_shadow_contract(client):
    resp = client.post("/v1/fulfillment/seller-sla-shadow", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction_mode"] == "SHADOW"
    assert body["automated_action_taken"] is False
    assert body["model_name"] == "olist_production_parity_seller_sla"
    assert 0.0 <= body["seller_sla_breach_probability"] <= 1.0


def test_route_rejects_negative_price(client):
    resp = client.post("/v1/fulfillment/seller-sla-shadow", json=_payload(total_price=-1.0))
    assert resp.status_code == 422


def test_route_rejects_missing_field(client):
    payload = _payload()
    del payload["same_zone"]
    resp = client.post("/v1/fulfillment/seller-sla-shadow", json=payload)
    assert resp.status_code == 422


def test_route_requires_auth():
    with TestClient(app) as c:
        resp = c.post("/v1/fulfillment/seller-sla-shadow", json=_payload())
    assert resp.status_code in (401, 403)


def test_route_persists_prediction(tmp_path):
    from src.ai_service.services.prediction_feedback_store import PredictionFeedbackStore

    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        app.state.seller_sla_feedback_store = PredictionFeedbackStore(path=tmp_path / "shadow_preds.jsonl")
        resp = c.post("/v1/fulfillment/seller-sla-shadow", json=_payload(order_ref="order_shadow_persist"))
        assert resp.status_code == 200
        assert resp.json()["persisted"] is True
    import json

    lines = (tmp_path / "shadow_preds.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["order_id"] == "order_shadow_persist"
    assert parsed["prediction_type"] == "seller_sla_breach"


def test_ready_endpoint_reports_shadow_model_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"]["production_parity_seller_sla_shadow_model_loaded"] is True


def test_research_and_shadow_routes_are_distinct(client):
    # Regression guard for Gate 10: the two routes must never collapse into
    # one contract. Research route requires seller_history_available;
    # shadow route requires same_zone. Sending the wrong payload shape to
    # each must fail validation, not silently coerce.
    research_payload_sent_to_shadow = {
        "order_ref": "x",
        "purchase_timestamp": "2024-01-01T00:00:00",
        "seller_history_available": False,
        "days_to_shipping_deadline": 1.0,
        "n_items": 1, "n_distinct_products": 1, "n_categories": 1,
        "total_price": 1.0, "total_freight": 1.0, "weight_g": 1.0, "volume_cm3": 1.0,
        "payment_value": 1.0, "n_installments": 1, "same_state": False,
    }
    resp = client.post("/v1/fulfillment/seller-sla-shadow", json=research_payload_sent_to_shadow)
    assert resp.status_code == 422  # missing same_zone, the shadow route's own required field
