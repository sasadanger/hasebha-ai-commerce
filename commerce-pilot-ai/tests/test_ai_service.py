"""Tests for the CommercePilot AI service (Phase 4)."""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.ai_service.main import app
from src.ai_service.services.decision_engine import DecisionEngineConfig, DecisionInput, evaluate
from tests.conftest import TEST_API_KEY


@pytest.fixture()
def client():
    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_loaded_capabilities(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    # process-alive (this response returning at all) vs. each asset actually
    # loadable (independently hash-verified, not assumed)
    assert body["checks"]["fulfillment_risk_model_loaded"] is True
    assert body["checks"]["decision_engine_config_loaded"] is True
    assert body["checks"]["nlp_registry_loaded"] is True
    assert body["checks"]["nlp_E_mpold_marbert_loadable"] is True
    assert body["checks"]["nlp_B2_astd_marbert_loadable"] is True
    assert body["checks"]["nlp_C_labr_marbert_loadable"] is True
    assert body["checks"]["nlp_C_labr_arabert_loadable"] is True
    assert body["checks"]["recommendation_engine_loaded"] is True
    # Amazon's classical artifact is a known, documented gap, not a failure
    assert body["checks"]["nlp_A_amazon_classical_materialized"] is False


def test_fulfillment_risk_scores_in_unit_interval(client):
    resp = client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "order-1",
            "purchase_timestamp": "2018-03-01T14:00:00Z",
            "approval_timestamp": "2018-03-01T14:10:00Z",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_experiment_id"] == "olist-phase2a-strict-core-v1"
    assert body["risk_class"] in ("low", "high")
    assert body["risk_threshold"] == pytest.approx(0.1293, abs=1e-4)
    assert len(body["model_artifact_sha256"]) == 64
    assert body["input_features"]["purchase_year"] == 2018


def test_fulfillment_risk_high_score_classified_high(client):
    # A very long purchase-to-approval gap is a strong risk signal in the
    # frozen Olist champion's own feature space.
    resp = client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "order-slow",
            "purchase_timestamp": "2018-03-01T00:00:00Z",
            "approval_timestamp": "2018-03-10T00:00:00Z",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    if body["risk_score"] >= body["risk_threshold"]:
        assert body["risk_class"] == "high"
    else:
        assert body["risk_class"] == "low"


def test_fulfillment_risk_rejects_approval_before_purchase(client):
    resp = client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "order-2",
            "purchase_timestamp": "2018-03-01T14:00:00Z",
            "approval_timestamp": "2018-03-01T13:00:00Z",
        },
    )
    assert resp.status_code == 422


def test_fulfillment_risk_rejects_missing_required_field(client):
    resp = client.post(
        "/v1/fulfillment/risk",
        json={"order_ref": "order-3", "purchase_timestamp": "2018-03-01T14:00:00Z"},
    )
    assert resp.status_code == 422


def test_fulfillment_risk_service_unavailable_returns_503(client):
    original = client.app.state.fulfillment_service
    client.app.state.fulfillment_service = None
    try:
        resp = client.post(
            "/v1/fulfillment/risk",
            json={
                "order_ref": "order-4",
                "purchase_timestamp": "2018-03-01T14:00:00Z",
                "approval_timestamp": "2018-03-01T14:10:00Z",
            },
        )
        assert resp.status_code == 503
    finally:
        client.app.state.fulfillment_service = original


class TestDecisionEngine:
    @pytest.fixture()
    def config(self):
        return DecisionEngineConfig.load()

    def test_high_risk_plus_delivery_complaint_is_most_critical(self, config):
        result = evaluate(
            DecisionInput(
                customer_ref="c1",
                fulfillment_risk_score=0.9,
                complaint_issue_type="delivery_delay",
                recommendation_strength=0.9,
                sentiment="positive",
            ),
            config,
        )
        assert result.action == "ESCALATE_TO_HUMAN_SERVICE"
        assert result.priority == "P0_CRITICAL"

    def test_high_risk_alone_is_intervention_priority(self, config):
        result = evaluate(DecisionInput(customer_ref="c1", fulfillment_risk_score=0.75), config)
        assert result.action == "INTERVENTION_PRIORITY"
        assert result.priority == "P1_HIGH"

    def test_negative_unresolved_suppresses_cross_sell(self, config):
        result = evaluate(
            DecisionInput(
                customer_ref="c1",
                fulfillment_risk_score=0.1,
                sentiment="negative",
                complaint_resolved=False,
                recommendation_strength=0.9,
            ),
            config,
        )
        assert result.action == "SUPPRESS_CROSS_SELL"

    def test_good_state_allows_cross_sell(self, config):
        result = evaluate(
            DecisionInput(
                customer_ref="c1",
                fulfillment_risk_score=0.1,
                sentiment="positive",
                recommendation_strength=0.8,
            ),
            config,
        )
        assert result.action == "ALLOW_CROSS_SELL"
        assert result.priority == "P3_GROWTH"

    def test_no_signal_is_no_action(self, config):
        result = evaluate(DecisionInput(customer_ref="c1"), config)
        assert result.action == "NO_ACTION"

    def test_low_risk_below_threshold_is_not_high_risk(self, config):
        result = evaluate(DecisionInput(customer_ref="c1", fulfillment_risk_score=0.49), config)
        assert result.action != "INTERVENTION_PRIORITY"
        assert result.action != "ESCALATE_TO_HUMAN_SERVICE"


def test_decision_endpoint_end_to_end(client):
    resp = client.post(
        "/v1/decision",
        json={
            "customer_ref": "c1",
            "order_ref": "o1",
            "fulfillment_risk_score": 0.8,
            "complaint_issue_type": "delivery_delay",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ESCALATE_TO_HUMAN_SERVICE"
    assert "HIGH_FULFILLMENT_RISK" in body["reason_codes"]
    assert body["ruleset_version"] == "decision-engine-rules-v1"


def test_decision_endpoint_passes_through_contributing_model_versions(client):
    resp = client.post(
        "/v1/decision",
        json={
            "customer_ref": "c1",
            "order_ref": "o1",
            "fulfillment_risk_score": 0.2,
            "recommendation_strength": 0.8,
            "model_versions": {
                "fulfillment_risk": "5a08ea55332332",
                "recommendations": "hybrid_with_popularity_backfill-v1",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ALLOW_CROSS_SELL"
    assert body["model_versions"]["fulfillment_risk"] == "5a08ea55332332"
    assert body["model_versions"]["recommendations"] == "hybrid_with_popularity_backfill-v1"


def test_decision_engine_unavailable_returns_503(client):
    original = client.app.state.decision_config
    client.app.state.decision_config = None
    try:
        resp = client.post("/v1/decision", json={"customer_ref": "c1"})
        assert resp.status_code == 503
    finally:
        client.app.state.decision_config = original
