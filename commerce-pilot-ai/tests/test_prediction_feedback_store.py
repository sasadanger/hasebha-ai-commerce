"""Tests for the minimal append-only seller-SLA prediction/outcome feedback store."""
import json

from src.ai_service.services.prediction_feedback_store import PredictionFeedbackStore


def test_record_prediction_appends_valid_json_line(tmp_path):
    store = PredictionFeedbackStore(path=tmp_path / "preds.jsonl")
    rec = store.record_prediction(
        order_id="o1", seller_id="s1", prediction_stage="T0", features_as_of="2024-01-01T00:00:00",
        score=0.42, risk_level="MEDIUM", model_name="olist_seller_sla", model_version="olist_seller_sla_v1",
        artifact_sha256="abc123", cold_start=False,
    )
    lines = (tmp_path / "preds.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["prediction_id"] == rec.prediction_id
    assert parsed["score"] == 0.42


def test_record_outcome_links_by_prediction_id(tmp_path):
    store = PredictionFeedbackStore(path=tmp_path / "preds.jsonl")
    rec = store.record_prediction(
        order_id="o1", seller_id="s1", prediction_stage="T0", features_as_of="2024-01-01T00:00:00",
        score=0.1, risk_level="LOW", model_name="olist_seller_sla", model_version="olist_seller_sla_v1",
        artifact_sha256="abc123", cold_start=True,
    )
    ok = store.record_outcome(rec.prediction_id, seller_sla_breached_actual=True)
    assert ok is True
    all_records = store.read_all()
    assert len(all_records) == 2
    outcome_row = [r for r in all_records if r.get("outcome") is not None][0]
    assert outcome_row["prediction_id"] == rec.prediction_id
    assert outcome_row["outcome"]["seller_sla_breached_actual"] is True


def test_record_outcome_on_empty_store_returns_false(tmp_path):
    store = PredictionFeedbackStore(path=tmp_path / "does_not_exist.jsonl")
    assert store.record_outcome("nonexistent-id") is False


def test_record_prediction_persists_raw_features(tmp_path):
    store = PredictionFeedbackStore(path=tmp_path / "preds.jsonl")
    raw = {"n_items": 3, "total_price": 199.99, "same_state": True}
    rec = store.record_prediction(
        order_id="o1", seller_id="s1", prediction_stage="T0", features_as_of="2024-01-01T00:00:00",
        score=0.42, risk_level="MEDIUM", model_name="olist_seller_sla", model_version="olist_seller_sla_v1",
        artifact_sha256="abc123", cold_start=False, raw_features=raw, feature_schema_version="test_v1",
    )
    assert rec.raw_features == raw
    assert rec.feature_schema_version == "test_v1"
    parsed = json.loads((tmp_path / "preds.jsonl").read_text().strip())
    assert parsed["raw_features"] == raw
    assert parsed["feature_schema_version"] == "test_v1"


def test_record_prediction_without_raw_features_defaults_to_none_backward_compatible(tmp_path):
    # A caller that does not pass raw_features (e.g. old code, or a future
    # caller that legitimately has none) must still produce a valid record --
    # this field is additive, never required.
    store = PredictionFeedbackStore(path=tmp_path / "preds.jsonl")
    rec = store.record_prediction(
        order_id="o1", seller_id="s1", prediction_stage="T0", features_as_of="2024-01-01T00:00:00",
        score=0.1, risk_level="LOW", model_name="olist_seller_sla", model_version="olist_seller_sla_v1",
        artifact_sha256="abc123", cold_start=False,
    )
    assert rec.raw_features is None
    assert rec.feature_schema_version is None


def test_read_all_tolerates_pre_existing_rows_without_raw_features_field(tmp_path):
    # Simulates rows written by the PREVIOUS version of this store (before
    # raw_features existed) sitting alongside new rows in the same file --
    # must not raise, must remain readable.
    path = tmp_path / "preds.jsonl"
    old_style_row = {
        "prediction_id": "old-1", "order_id": "o0", "seller_id": None,
        "prediction_type": "seller_sla_breach", "prediction_stage": "T0",
        "prediction_timestamp": "2026-08-15T00:00:00+00:00", "features_as_of": "2026-08-15T00:00:00",
        "score": 0.2, "risk_level": "LOW", "model_name": "olist_seller_sla",
        "model_version": "olist_seller_sla_v1", "artifact_sha256": "abc", "cold_start": False,
        "decision": None, "outcome": None,
    }
    path.write_text(json.dumps(old_style_row) + "\n")
    store = PredictionFeedbackStore(path=path)
    store.record_prediction(
        order_id="o1", seller_id="s1", prediction_stage="T0", features_as_of="2024-01-01T00:00:00",
        score=0.3, risk_level="LOW", model_name="olist_seller_sla", model_version="olist_seller_sla_v1",
        artifact_sha256="abc123", cold_start=False, raw_features={"x": 1},
    )
    rows = store.read_all()
    assert len(rows) == 2
    assert rows[0]["prediction_id"] == "old-1"
    assert "raw_features" not in rows[0] or rows[0]["raw_features"] is None
    assert rows[1]["raw_features"] == {"x": 1}


def test_route_scoring_persists_a_prediction_record(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.ai_service.main import app
    from tests.conftest import TEST_API_KEY

    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as client:
        from src.ai_service.services.prediction_feedback_store import PredictionFeedbackStore
        app.state.seller_sla_feedback_store = PredictionFeedbackStore(path=tmp_path / "route_preds.jsonl")
        payload = {
            "order_ref": "order_persist_test",
            "purchase_timestamp": "2024-03-15T10:00:00",
            "seller_history_available": False,
            "days_to_shipping_deadline": 5.0,
            "n_items": 1, "n_distinct_products": 1, "n_categories": 1,
            "total_price": 80.0, "total_freight": 10.0, "weight_g": 300.0, "volume_cm3": 1200.0,
            "payment_value": 90.0, "n_installments": 1, "same_state": False,
        }
        resp = client.post("/v1/fulfillment/seller-sla-risk", json=payload)
        assert resp.status_code == 200
    persisted = (tmp_path / "route_preds.jsonl").read_text().strip().splitlines()
    assert len(persisted) == 1
    parsed = json.loads(persisted[0])
    assert parsed["order_id"] == "order_persist_test"
    assert parsed["raw_features"] is not None
    assert parsed["raw_features"]["order_ref"] == "order_persist_test"
    assert parsed["raw_features"]["total_price"] == 80.0
    assert parsed["feature_schema_version"] == "seller_sla_risk_request_v1"


def test_shadow_route_persists_raw_features(tmp_path):
    from fastapi.testclient import TestClient
    from src.ai_service.main import app
    from tests.conftest import TEST_API_KEY

    with TestClient(app, headers={"X-Internal-Api-Key": TEST_API_KEY}) as client:
        from src.ai_service.services.prediction_feedback_store import PredictionFeedbackStore
        app.state.seller_sla_feedback_store = PredictionFeedbackStore(path=tmp_path / "shadow_preds.jsonl")
        payload = {
            "order_ref": "order_shadow_persist_test",
            "purchase_timestamp": "2024-03-15T10:00:00",
            "n_items": 2, "n_distinct_products": 2, "n_categories": 1,
            "total_price": 150.0, "total_freight": 20.0, "weight_g": 800.0, "volume_cm3": 4000.0,
            "payment_value": 170.0, "same_zone": False,
        }
        resp = client.post("/v1/fulfillment/seller-sla-shadow", json=payload)
        assert resp.status_code == 200
    persisted = (tmp_path / "shadow_preds.jsonl").read_text().strip().splitlines()
    assert len(persisted) == 1
    parsed = json.loads(persisted[0])
    assert parsed["raw_features"]["order_ref"] == "order_shadow_persist_test"
    assert parsed["raw_features"]["total_price"] == 150.0
    assert parsed["feature_schema_version"] == "production_parity_seller_sla_shadow_request_v1"
