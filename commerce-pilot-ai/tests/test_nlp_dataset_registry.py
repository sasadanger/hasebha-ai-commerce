import json
import yaml

REGISTRY_JSON = "reports/generated/nlp/dataset_registry.json"
REGISTRY_YAML = "configs/nlp_dataset_registry.yaml"

VALID_APPROVAL = {
    "APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT",
    "APPROVED_FOR_BENCHMARK_ONLY",
    "QUARANTINE_LICENSE",
    "QUARANTINE_PROVENANCE",
    "QUARANTINE_PRIVACY",
    "QUARANTINE_CONFLICTING_METADATA",
    "ACCESS_PENDING",
    "DATA_NOT_PUBLICLY_AVAILABLE",
    "REJECTED",
}
APPROVED_STATES = {"APPROVED_FOR_FUTURE_RESEARCH_EXPERIMENT", "APPROVED_FOR_BENCHMARK_ONLY"}


def load_registry():
    with open(REGISTRY_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_registry_yaml():
    with open(REGISTRY_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_registry_is_machine_readable():
    r = load_registry()
    assert r["schema_version"] == "nlp-dataset-registry-v1"
    assert r["dataset_count"] == len(r["datasets"])


def test_yaml_and_json_registries_are_consistent():
    rj = load_registry()
    ry = load_registry_yaml()
    assert rj["dataset_count"] == ry["dataset_count"]
    ids_j = {d["dataset_id"] for d in rj["datasets"]}
    ids_y = {d["dataset_id"] for d in ry["datasets"]}
    assert ids_j == ids_y


def test_every_dataset_id_is_unique():
    r = load_registry()
    ids = [d["dataset_id"] for d in r["datasets"]]
    assert len(ids) == len(set(ids))


def test_approval_status_is_valid_for_every_dataset():
    r = load_registry()
    for d in r["datasets"]:
        assert d["approval_status"] in VALID_APPROVAL, d["dataset_id"]


def test_approved_datasets_have_provenance_evidence():
    r = load_registry()
    for d in r["datasets"]:
        if d["approval_status"] in APPROVED_STATES:
            assert d.get("evidence_sources"), d["dataset_id"]
            assert len(d["evidence_sources"]) > 0


def test_approved_datasets_have_source_url_or_doi():
    r = load_registry()
    for d in r["datasets"]:
        if d["approval_status"] in APPROVED_STATES:
            assert d.get("primary_source_url") or d.get("doi") or d.get("repository_url"), d["dataset_id"]


def test_downloaded_datasets_have_sha256_hashes():
    r = load_registry()
    for d in r["datasets"]:
        if d.get("download_status") == "DOWNLOADED_THIS_SESSION":
            manifest = d.get("extracted_hash_manifest") or {}
            assert manifest, d["dataset_id"]
            for fname, h in manifest.items():
                assert h is not None and len(h) == 64, f"{d['dataset_id']}/{fname}"


def test_quarantined_datasets_are_not_marked_approved():
    r = load_registry()
    for d in r["datasets"]:
        if str(d.get("approval_status", "")).startswith("QUARANTINE"):
            assert d["approval_status"] not in APPROVED_STATES


def test_no_protected_test_path_appears_in_registry():
    r = load_registry()
    blob = json.dumps(r).lower()
    assert "phase2a_test" not in blob
    assert "test_access_ledger" not in blob
    assert "olist-phase2a" not in blob


def test_amazon_appliances_remains_preserved():
    r = load_registry()
    ids = {d["dataset_id"] for d in r["datasets"]}
    assert "amazon_appliances" in ids
    amz = next(d for d in r["datasets"] if d["dataset_id"] == "amazon_appliances")
    assert amz["approval_status"] in APPROVED_STATES


def test_registry_does_not_claim_egyptian_readiness():
    r = load_registry()
    blob = json.dumps(r)
    assert "EGYPTIAN_MARKET_READINESS = PROVEN" not in blob
    assert '"egyptian_market_readiness": "PROVEN"' not in blob


def test_no_dataset_falsely_labeled_egyptian_ecommerce_external_validation():
    r = load_registry()
    for d in r["datasets"]:
        roles = d.get("recommended_role", [])
        if "EGYPTIAN_EXTERNAL_VALIDATION_CANDIDATE" in " ".join(roles):
            # only the Egyptian Tweets 40K entry carries this tentative role, and it is explicitly marked tentative
            assert d["dataset_id"] == "egyptian_tweets_corpus_40k"


def test_eesa_not_substituted_with_unrelated_dataset():
    r = load_registry()
    eesa = next(d for d in r["datasets"] if d["dataset_id"] == "eesa_named_dataset")
    assert eesa["download_status"] == "DATASET_NOT_IDENTIFIED_UNDER_THIS_NAME"
    assert eesa["approval_status"] == "DATA_NOT_PUBLICLY_AVAILABLE"
