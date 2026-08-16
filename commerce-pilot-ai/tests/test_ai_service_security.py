"""Regression tests for the Security Release Gate (2026-08-16) fixes.

Covers:
- CWE-306 / OWASP API2:2023 (missing authentication): the inference
  endpoints must reject requests without a valid X-Internal-Api-Key, while
  /health and /ready must remain open (platform health checks have no
  credential).
- Resource-exhaustion hardening: the NLP and recommendation endpoints must
  reject oversized inputs rather than accept unbounded payloads.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.ai_service.main import app
from tests.conftest import TEST_API_KEY


@pytest.fixture()
def unauthenticated_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authenticated_client():
    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        yield c


# --- Missing authentication (CWE-306) ---------------------------------------

@pytest.mark.parametrize(
    "method,path,json_body",
    [
        (
            "post",
            "/v1/fulfillment/risk",
            {
                "order_ref": "o1",
                "purchase_timestamp": "2018-03-01T14:00:00Z",
                "approval_timestamp": "2018-03-01T14:10:00Z",
            },
        ),
        ("post", "/v1/decision", {"customer_ref": "c1"}),
        ("post", "/v1/nlp/analyze", {"text": "hello", "task": "E"}),
        ("post", "/v1/recommendations", {"customer_ref": "c1"}),
    ],
)
def test_inference_endpoints_reject_missing_api_key(unauthenticated_client, method, path, json_body):
    resp = getattr(unauthenticated_client, method)(path, json=json_body)
    assert resp.status_code == 401
    assert "X-Internal-Api-Key" in resp.json()["detail"]


def test_inference_endpoint_rejects_wrong_api_key(unauthenticated_client):
    resp = unauthenticated_client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "o1",
            "purchase_timestamp": "2018-03-01T14:00:00Z",
            "approval_timestamp": "2018-03-01T14:10:00Z",
        },
        headers={"X-Internal-Api-Key": "definitely-not-the-real-key"},
    )
    assert resp.status_code == 401


def test_health_and_ready_remain_public(unauthenticated_client):
    # Platform health checks (Railway, uptime monitors) have no credential
    # and must never be gated behind the API key.
    assert unauthenticated_client.get("/health").status_code == 200
    assert unauthenticated_client.get("/ready").status_code == 200


def test_inference_endpoint_succeeds_with_correct_api_key(authenticated_client):
    resp = authenticated_client.post(
        "/v1/fulfillment/risk",
        json={
            "order_ref": "o1",
            "purchase_timestamp": "2018-03-01T14:00:00Z",
            "approval_timestamp": "2018-03-01T14:10:00Z",
        },
    )
    assert resp.status_code == 200


# --- Resource-exhaustion hardening ------------------------------------------

def test_nlp_analyze_rejects_oversized_text(authenticated_client):
    resp = authenticated_client.post(
        "/v1/nlp/analyze",
        json={"text": "a" * 10_001, "task": "E"},
    )
    assert resp.status_code == 422


def test_nlp_analyze_accepts_text_at_the_limit(authenticated_client):
    # Boundary check: exactly at the limit must not be rejected by the
    # length constraint itself (a real model/task error is fine here, but
    # not a 422 whose cause is the text length).
    resp = authenticated_client.post(
        "/v1/nlp/analyze",
        json={"text": "a" * 10_000, "task": "E"},
    )
    assert resp.status_code != 422 or "text" not in str(
        [e.get("loc") for e in resp.json().get("detail", [])]
    )


def test_recommendations_rejects_oversized_history(authenticated_client):
    oversized_history = [
        {"item_id": f"item-{i}", "times_purchased": 1, "orders_since_last_purchase": 0}
        for i in range(1001)
    ]
    resp = authenticated_client.post(
        "/v1/recommendations",
        json={"customer_ref": "c1", "history": oversized_history},
    )
    assert resp.status_code == 422


def test_recommendations_rejects_oversized_k(authenticated_client):
    resp = authenticated_client.post(
        "/v1/recommendations",
        json={"customer_ref": "c1", "k": 100_000},
    )
    assert resp.status_code == 422
