"""Tests for the real NLP and recommendation inference adapters (REAL AI
INFERENCE INTEGRATION phase). Split from test_ai_service.py because these
exercises load actual fine-tuned transformer weights and are materially
slower than the rest of the suite.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.ai_service.main import app
from src.ai_service.services.nlp_inference import ArtifactIntegrityError, NlpInferenceService
from src.ai_service.services.recommendation_engine import RecommendationEngineService
from tests.conftest import TEST_API_KEY


@pytest.fixture(scope="module")
def client():
    # Module-scoped deliberately: transformer weights load lazily and are
    # cached per NlpInferenceService instance, so reusing one app/service
    # instance across this file's tests means each of the 4 real models is
    # loaded at most once for the whole file, not once per test -- keeping
    # peak RAM bounded per the project's "one heavy workload at a time"
    # resource discipline.
    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        yield c


# --- NLP: task/model routing -------------------------------------------------


def test_nlp_analyze_mpold_uses_promoted_marbert(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "This product is terrible.", "task": "E"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"
    assert body["task_name"] == "MPOLD"
    assert body["champion_selected"] is True
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["model_name"] == "UBC-NLP/MARBERT"
    assert pred["model_revision"] == "88e1fa192dd723cf0b3563500aec46209762eb22"
    assert pred["predicted_label"] in ("Offensive", "Non-Offensive")
    assert abs(sum(pred["class_probabilities"].values()) - 1.0) < 1e-4


def test_nlp_analyze_astd_uses_promoted_marbert(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "منتج جيد جدا", "task": "B2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion_selected"] is True
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_label"] in ("NEG", "NEUTRAL", "OBJ", "POS")


def test_nlp_analyze_labr_returns_both_finalists_unresolved(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "كتاب جيد", "task": "C"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["champion_selected"] is False
    keys = {p["model_key"] for p in body["predictions"]}
    assert keys == {"C_MARBERT", "C_AraBERT"}


def test_nlp_analyze_labr_can_pin_one_finalist(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "كتاب جيد", "task": "C", "model": "arabert"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["model_key"] == "C_AraBERT"


def test_nlp_analyze_amazon_is_honestly_not_materialized(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "great appliance", "task": "A"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ARTIFACT_NOT_MATERIALIZED"
    assert "winner.json" in body["frozen_config_reference"]
    assert "training" in body["reason"].lower()


def test_nlp_analyze_rejects_unknown_task(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "x", "task": "Z"})
    assert resp.status_code == 422


def test_nlp_analyze_rejects_unknown_model_choice(client):
    resp = client.post("/v1/nlp/analyze", json={"text": "x", "task": "C", "model": "gpt-nonsense"})
    assert resp.status_code == 422


def test_nlp_analyze_service_unavailable_returns_503(client):
    original = client.app.state.nlp_service
    client.app.state.nlp_service = None
    try:
        resp = client.post("/v1/nlp/analyze", json={"text": "x", "task": "E"})
        assert resp.status_code == 503
    finally:
        client.app.state.nlp_service = original


# --- NLP: loader/integrity failure path (no real weights needed) ------------


def test_nlp_service_refuses_to_construct_with_missing_export_file(tmp_path):
    manifest = {
        "schema_version": "nlp-inference-export-manifest-v1",
        "exported_at": "test",
        "exports": [
            {
                "key": "FAKE",
                "task": "E",
                "task_name": "MPOLD",
                "model_name": "fake/model",
                "revision": "deadbeef",
                "max_length": 128,
                "labels": ["a", "b"],
                "export_dir": "does/not/exist",
                "model_file_hashes": {"config.json": "0" * 64},
                "tokenizer_file_hashes": {},
            }
        ],
    }
    manifest_path = tmp_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        NlpInferenceService(manifest_path=manifest_path)


def test_nlp_service_refuses_to_construct_on_hash_mismatch(tmp_path):
    export_dir = tmp_path / "FAKE"
    export_dir.mkdir()
    (export_dir / "config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": "nlp-inference-export-manifest-v1",
        "exported_at": "test",
        "exports": [
            {
                "key": "FAKE",
                "task": "E",
                "task_name": "MPOLD",
                "model_name": "fake/model",
                "revision": "deadbeef",
                "max_length": 128,
                "labels": ["a", "b"],
                "export_dir": str(export_dir),
                "model_file_hashes": {"config.json": "0" * 64},
                "tokenizer_file_hashes": {},
            }
        ],
    }
    manifest_path = tmp_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ArtifactIntegrityError, FileNotFoundError)):
        NlpInferenceService(manifest_path=manifest_path)


# --- Recommendations ---------------------------------------------------------


HISTORY = [
    {"item_id": "sku-milk", "times_purchased": 8, "orders_since_last_purchase": 0},
    {"item_id": "sku-bread", "times_purchased": 5, "orders_since_last_purchase": 1},
    {"item_id": "sku-eggs", "times_purchased": 2, "orders_since_last_purchase": 6},
]


def test_recommendation_engine_is_deterministic():
    service = RecommendationEngineService()
    r1 = service.recommend(history=HISTORY, k=10)
    r2 = service.recommend(history=HISTORY, k=10)
    ids1 = [i.item_id for i in r1.items]
    ids2 = [i.item_id for i in r2.items]
    assert ids1 == ids2


def test_recommendation_reorder_ranks_recent_frequent_items_first():
    service = RecommendationEngineService()
    result = service.recommend(history=HISTORY, k=10)
    reorder_items = [i for i in result.items if i.component == "reorder"]
    reorder_ids = [i.item_id for i in reorder_items]
    # sku-milk: freq 8, gap 0 (most recent+most frequent) should rank first;
    # sku-eggs: freq 2, gap 6 (least recent+least frequent) should rank last
    # among the reorder slots.
    assert reorder_ids[0] == "sku-milk"
    assert reorder_ids[-1] == "sku-eggs"
    assert len(reorder_items) == 3


def test_recommendation_endpoint_returns_ranked_items(client):
    resp = client.post(
        "/v1/recommendations",
        json={"customer_ref": "cust-1", "k": 10, "history": HISTORY},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["k"] == 10
    assert len(body["items"]) == 10
    assert [i["rank"] for i in body["items"]] == list(range(1, 11))
    reorder = [i for i in body["items"] if i["component"] == "reorder"]
    backfill = [i for i in body["items"] if i["component"] == "discovery_backfill"]
    assert len(reorder) == 3
    assert len(backfill) == 7
    assert all(i["cross_catalog_deployment_status"] == "VALID_ANY_CATALOG" for i in reorder)
    assert all(i["cross_catalog_deployment_status"] == "NOT_YET_VALIDATED" for i in backfill)
    assert body["egyptian_catalog_mapping_status"] == "NOT_YET_VALIDATED"
    assert len(body["ranking_logic_sha256"]) == 64


def test_recommendation_empty_history_is_backfill_only(client):
    resp = client.post("/v1/recommendations", json={"customer_ref": "cold-start", "k": 5, "history": []})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 5
    assert all(i["component"] == "discovery_backfill" for i in body["items"])


def test_recommendation_unrecognized_item_id_does_not_crash(client):
    # A history item_id with no relationship to the Instacart catalog is
    # still valid input -- the generic contract does not require catalog
    # membership for reorder candidates, only that the caller vouches the
    # customer bought it before.
    resp = client.post(
        "/v1/recommendations",
        json={
            "customer_ref": "c2",
            "k": 5,
            "history": [{"item_id": "not-a-real-instacart-sku-xyz", "times_purchased": 1, "orders_since_last_purchase": 0}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    reorder_ids = [i["item_id"] for i in body["items"] if i["component"] == "reorder"]
    assert "not-a-real-instacart-sku-xyz" in reorder_ids


def test_recommendation_rejects_k_outside_evidenced_operating_points(client):
    resp = client.post("/v1/recommendations", json={"customer_ref": "c3", "k": 7, "history": []})
    assert resp.status_code == 422


def test_recommendation_service_unavailable_returns_503(client):
    original = client.app.state.recommendation_service
    client.app.state.recommendation_service = None
    try:
        resp = client.post("/v1/recommendations", json={"customer_ref": "c4", "history": []})
        assert resp.status_code == 503
    finally:
        client.app.state.recommendation_service = original


def test_recommendation_engine_error_path_returns_500_not_a_crash():
    with TestClient(app, raise_server_exceptions=False, headers={"X-Internal-Api-Key": TEST_API_KEY}) as c:
        original = c.app.state.recommendation_service.recommend

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated model failure")

        c.app.state.recommendation_service.recommend = _boom
        try:
            resp = c.post("/v1/recommendations", json={"customer_ref": "c5", "history": []})
            assert resp.status_code == 500
        finally:
            c.app.state.recommendation_service.recommend = original
        # The app process itself is still alive and answering afterward.
        resp2 = c.get("/health")
        assert resp2.status_code == 200
