import json
import yaml

AUDIT_PATH = "reports/generated/olist/phase2c/asof_feature_audit.json"
CONTRACT_PATH = "configs/olist_phase2c_asof_feature_contract.yaml"

VALID_CLASSIFICATIONS = {
    "PROVEN_PRIMARY_ASOF",
    "UNVERIFIED_ASOF",
    "RETROSPECTIVE_ONLY",
    "FORBIDDEN_LEAKAGE",
    "IDENTIFIER_ONLY",
    "POST_OUTCOME",
    "UNKNOWN",
}
NON_PRIMARY_ELIGIBLE = {"POST_OUTCOME", "FORBIDDEN_LEAKAGE"}


def load_audit():
    with open(AUDIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_contract():
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_audit_is_machine_readable():
    audit = load_audit()
    assert isinstance(audit, dict)
    assert audit["schema_version"] == "olist-phase2c-asof-feature-audit-v1"


def test_all_15_features_appear_exactly_once():
    audit = load_audit()
    names = [r["feature_name"] for r in audit["feature_records"]]
    assert len(names) == 15
    assert len(set(names)) == 15


def test_every_feature_has_one_valid_classification():
    audit = load_audit()
    for r in audit["feature_records"]:
        assert r["final_classification"] in VALID_CLASSIFICATIONS


def test_primary_asof_only_when_classification_proven():
    audit = load_audit()
    for r in audit["feature_records"]:
        if r["primary_asof"]:
            assert r["final_classification"] == "PROVEN_PRIMARY_ASOF"


def test_no_post_outcome_or_forbidden_feature_is_primary_asof():
    audit = load_audit()
    for r in audit["feature_records"]:
        if r["final_classification"] in NON_PRIMARY_ELIGIBLE:
            assert r["primary_asof"] is False


def test_decision_timestamp_is_order_approved_at():
    audit = load_audit()
    assert audit["decision_timestamp"] == "order_approved_at"
    for r in audit["feature_records"]:
        assert r["decision_timestamp"] == "order_approved_at"


def test_proven_primary_asof_records_have_required_evidence_fields():
    audit = load_audit()
    for r in audit["feature_records"]:
        if r["final_classification"] == "PROVEN_PRIMARY_ASOF":
            assert r["source_event_timestamp"] not in (None, "", "unknown", "none")
            assert r["evidence_paths"]


def test_classification_counts_total_15():
    audit = load_audit()
    assert sum(audit["classification_counts"].values()) == 15


def test_no_test_route_in_audit_artifact():
    audit = load_audit()
    blob = json.dumps(audit).lower()
    assert "test_access" not in blob
    assert "phase2a_test" not in blob


def test_contract_yaml_matches_audit_feature_set():
    audit = load_audit()
    contract = load_contract()
    audit_names = {r["feature_name"] for r in audit["feature_records"]}
    contract_names = {f["feature_name"] for f in contract["features"]}
    assert audit_names == contract_names
    assert len(contract["features"]) == 15


def test_contract_classifications_match_audit():
    audit = load_audit()
    contract = load_contract()
    audit_by_name = {r["feature_name"]: r["final_classification"] for r in audit["feature_records"]}
    for f in contract["features"]:
        assert f["classification"] == audit_by_name[f["feature_name"]]


def test_zero_features_currently_proven_matches_phase2b_inherited_state():
    audit = load_audit()
    assert audit["classification_counts"]["PROVEN_PRIMARY_ASOF"] == 0
    assert audit["classification_counts"]["UNVERIFIED_ASOF"] == 15
