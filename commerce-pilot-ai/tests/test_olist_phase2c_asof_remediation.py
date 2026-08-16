import json

REMEDIATION_PATH = "reports/generated/olist/phase2c/asof_evidence_remediation.json"
AUDIT_PATH = "reports/generated/olist/phase2c/asof_feature_audit.json"

VALID_CLASSIFICATIONS = {
    "PROVEN_PRIMARY_ASOF",
    "UNVERIFIED_ASOF",
    "RETROSPECTIVE_ONLY",
    "FORBIDDEN_LEAKAGE",
    "IDENTIFIER_ONLY",
    "POST_OUTCOME",
    "UNKNOWN",
}
VALID_EVIDENCE_LEVEL_PREFIXES = ("LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL_5", "NONE")


def load_remediation():
    with open(REMEDIATION_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_audit():
    with open(AUDIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_remediation_is_machine_readable():
    r = load_remediation()
    assert r["schema_version"] == "olist-phase2c-asof-evidence-remediation-v1"


def test_all_15_features_appear_exactly_once():
    r = load_remediation()
    names = [rec["feature_name"] for rec in r["feature_records"]]
    assert len(names) == 15
    assert len(set(names)) == 15


def test_every_feature_has_valid_final_classification():
    r = load_remediation()
    for rec in r["feature_records"]:
        assert rec["final_classification"] in VALID_CLASSIFICATIONS


def test_primary_asof_only_when_classification_proven():
    r = load_remediation()
    for rec in r["feature_records"]:
        if rec["primary_asof"]:
            assert rec["final_classification"] == "PROVEN_PRIMARY_ASOF"


def test_promoted_features_have_direct_evidence():
    r = load_remediation()
    for rec in r["feature_records"]:
        if rec["final_classification"] == "PROVEN_PRIMARY_ASOF":
            assert rec["source_system_availability_proven"] is True
            assert rec["decision_time_availability_proven"] is True
            assert rec["revision_backfill_risk_resolved"] is True
            assert rec["evidence_level"].startswith(("LEVEL_1", "LEVEL_2", "LEVEL_3"))


def test_unresolved_features_are_not_promoted():
    r = load_remediation()
    for rec in r["feature_records"]:
        if rec["final_classification"] != "PROVEN_PRIMARY_ASOF":
            assert rec["primary_asof"] is False


def test_classification_counts_sum_to_15():
    r = load_remediation()
    assert sum(r["classification_counts_final"].values()) == 15
    assert sum(r["classification_counts_prior"].values()) == 15


def test_evidence_source_levels_are_valid():
    r = load_remediation()
    for s in r["external_evidence_sources"]:
        assert s["evidence_level"].startswith(VALID_EVIDENCE_LEVEL_PREFIXES)


def test_no_protected_test_path_introduced():
    r = load_remediation()
    blob = json.dumps(r).lower()
    assert "test_access" not in blob
    assert "phase2a_test" not in blob


def test_remediation_does_not_contradict_prior_audit_where_unchanged():
    r = load_remediation()
    audit = load_audit()
    audit_by_name = {rec["feature_name"]: rec["final_classification"] for rec in audit["feature_records"]}
    for rec in r["feature_records"]:
        if rec["final_classification"] == rec["prior_classification"]:
            assert audit_by_name[rec["feature_name"]] == rec["prior_classification"]


def test_outcome_is_one_of_three_allowed_values():
    r = load_remediation()
    assert r["asof_evidence_remediation"] in {
        "RESOLVED_FULLY",
        "RESOLVED_PARTIALLY",
        "IRREDUCIBLE_IN_OLIST",
    }


def test_zero_promotions_matches_irreducible_outcome():
    r = load_remediation()
    if r["asof_evidence_remediation"] == "IRREDUCIBLE_IN_OLIST":
        assert r["approved_expanded_primary_feature_count"] == 0
        assert r["classification_counts_final"]["PROVEN_PRIMARY_ASOF"] == 0
