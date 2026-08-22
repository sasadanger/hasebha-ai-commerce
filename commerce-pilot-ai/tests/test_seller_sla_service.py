"""Tests for SellerSlaRiskService and the /v1/fulfillment/seller-sla-risk route."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.ai_service.main import app
from src.ai_service.services.seller_sla_risk import SellerSlaRiskService, ModelIntegrityError
from tests.conftest import TEST_API_KEY


@pytest.fixture()
def client():
    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        yield c


def _known_seller_payload(**overrides):
    payload = {
        "order_ref": "order_123",
        "seller_ref": "seller_abc",
        "purchase_timestamp": "2024-03-15T10:00:00",
        "seller_history_available": True,
        "days_to_shipping_deadline": 3.0,
        "n_items": 2,
        "n_distinct_products": 2,
        "n_categories": 1,
        "total_price": 150.0,
        "total_freight": 20.0,
        "weight_g": 800.0,
        "volume_cm3": 4000.0,
        "payment_value": 170.0,
        "n_installments": 3,
        "same_state": True,
        "seller_past_order_count": 50,
        "seller_past_breach_rate_expanding": 0.08,
        "seller_past_handling_median_expanding": 2.0,
        "seller_past_handling_std_expanding": 1.0,
        "seller_breach_rate_30d": 0.05,
        "seller_breach_rate_90d": 0.07,
        "seller_handling_mean_30d": 2.1,
        "seller_recent_load_7d": 5.0,
    }
    payload.update(overrides)
    return payload


def _cold_start_payload(**overrides):
    payload = {
        "order_ref": "order_new",
        "purchase_timestamp": "2024-03-15T10:00:00",
        "seller_history_available": False,
        "days_to_shipping_deadline": 5.0,
        "n_items": 1,
        "n_distinct_products": 1,
        "n_categories": 1,
        "total_price": 80.0,
        "total_freight": 10.0,
        "weight_g": 300.0,
        "volume_cm3": 1200.0,
        "payment_value": 90.0,
        "n_installments": 1,
        "same_state": False,
    }
    payload.update(overrides)
    return payload


# --- Service-level tests -----------------------------------------------------


def test_service_loads_and_hash_matches():
    svc = SellerSlaRiskService()
    assert svc is not None


def test_known_seller_scoring_returns_valid_probability():
    svc = SellerSlaRiskService()

    class Req:
        pass

    req = Req()
    for k, v in _known_seller_payload().items():
        if k == "purchase_timestamp":
            v = datetime.fromisoformat(v)
        setattr(req, k, v)
    result = svc.score(req)
    assert 0.0 <= result.probability_calibrated <= 1.0
    assert result.cold_start is False
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}


def test_cold_start_uses_sentinel_not_fabricated_history():
    svc = SellerSlaRiskService()

    class Req:
        pass

    req = Req()
    payload = _cold_start_payload()
    for k, v in payload.items():
        if k == "purchase_timestamp":
            v = datetime.fromisoformat(v)
        setattr(req, k, v)
    for f in [
        "seller_past_order_count", "seller_past_breach_rate_expanding",
        "seller_past_handling_median_expanding", "seller_past_handling_std_expanding",
        "seller_breach_rate_30d", "seller_breach_rate_90d", "seller_handling_mean_30d",
        "seller_recent_load_7d",
    ]:
        setattr(req, f, None)
    result = svc.score(req)
    assert result.cold_start is True
    assert "COLD_START_NO_SELLER_HISTORY" in result.reason_codes


def test_missing_history_fields_when_flagged_available_raises():
    svc = SellerSlaRiskService()

    class Req:
        pass

    req = Req()
    payload = _known_seller_payload()
    payload["seller_past_order_count"] = None  # claims history available but omits a required field
    for k, v in payload.items():
        if k == "purchase_timestamp":
            v = datetime.fromisoformat(v)
        setattr(req, k, v)
    with pytest.raises(ValueError):
        svc.score(req)


# --- Route-level tests --------------------------------------------------------


def test_route_known_seller_returns_200(client):
    resp = client.post("/v1/fulfillment/seller-sla-risk", json=_known_seller_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_parity_status"] == "RESEARCH_OFFLINE_ONLY"
    assert 0.0 <= body["seller_sla_breach_probability"] <= 1.0
    assert body["cold_start"] is False
    assert "customer_late_probability" not in body


def test_route_cold_start_seller_returns_200(client):
    resp = client.post("/v1/fulfillment/seller-sla-risk", json=_cold_start_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["cold_start"] is True
    assert "COLD_START_NO_SELLER_HISTORY" in body["reason_codes"]


def test_route_history_available_but_missing_fields_422(client):
    payload = _known_seller_payload()
    payload["seller_past_order_count"] = None
    resp = client.post("/v1/fulfillment/seller-sla-risk", json=payload)
    assert resp.status_code == 422


def test_route_invalid_negative_price_rejected(client):
    payload = _known_seller_payload(total_price=-5.0)
    resp = client.post("/v1/fulfillment/seller-sla-risk", json=payload)
    assert resp.status_code == 422


def test_route_missing_required_field_rejected(client):
    payload = _known_seller_payload()
    del payload["purchase_timestamp"]
    resp = client.post("/v1/fulfillment/seller-sla-risk", json=payload)
    assert resp.status_code == 422


def test_route_requires_auth():
    with TestClient(app) as c:
        resp = c.post("/v1/fulfillment/seller-sla-risk", json=_known_seller_payload())
    assert resp.status_code in (401, 403)


def test_existing_v1_fulfillment_risk_route_still_present(client):
    # Backward-compatibility guard: this new route must never have displaced
    # the existing V1 endpoint.
    resp = client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "o1",
            "purchase_timestamp": "2024-01-01T00:00:00",
            "approval_timestamp": "2024-01-01T01:00:00",
        },
    )
    assert resp.status_code == 200
    assert "model_experiment_id" in resp.json()


def test_ready_endpoint_reports_seller_sla_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"]["seller_sla_risk_model_loaded"] is True


def test_service_rejects_tampered_artifact(tmp_path, monkeypatch):
    import src.ai_service.services.seller_sla_risk as mod

    fake_model = tmp_path / "fake.txt"
    fake_model.write_text("not a real model")
    # Patch the name directly in the already-imported module (no reload
    # needed/wanted -- reloading would leave the module object in sys.modules
    # permanently repointed for the rest of this test process, corrupting
    # every other test that imports SellerSlaRiskService afterward).
    # monkeypatch auto-reverts this attribute after the test.
    monkeypatch.setattr(mod, "SELLER_SLA_MODEL_PATH", fake_model)
    with pytest.raises(ModelIntegrityError):
        mod.SellerSlaRiskService()
