from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path):
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_amazon_contract_has_one_canonical_overall_field():
    manifest = load_yaml("configs/nlp_experiment_manifest.yaml")
    amazon = next(x for x in manifest["experiments"] if x["dataset"] == "amazon_appliances")
    assert amazon["label_fields"] == ["overall"]
    assert load_yaml("configs/nlp_label_ontology_v2.yaml")["source_native_concepts"]["AMAZON_APPLIANCES_RATING"]["source_field"] == "overall"
    assert load_yaml("configs/nlp_metric_contract_v2.yaml")["batch1_experiments"][amazon["experiment_id"]]["source_label"] == "overall"
    assert load_yaml("configs/nlp_split_policy.yaml")["experiment_splits"][amazon["experiment_id"]]["stratify_by"] == "overall"


def test_active_manifest_has_one_v2_split_policy_and_classical_batch1_only():
    manifest = load_yaml("configs/nlp_experiment_manifest.yaml")
    expected = {"A", "B2", "C", "E"}
    assert manifest["active_split_policy_version"] == "nlp-split-policy-v2"
    assert {x["batch_id"] for x in manifest["experiments"]} == expected
    assert {x["split_contract"].split("#")[0] for x in manifest["experiments"]} == {manifest["active_split_policy"]}
    prohibited = ("bert", "arabert", "marbert", "transformer", "xlm", "roberta", "distilbert")
    text = (ROOT / "configs/nlp_experiment_manifest.yaml").read_text(encoding="utf-8").lower()
    assert not any(term in text for term in prohibited)


def test_execution_configuration_defaults_are_complete_and_explicit():
    authorization = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")
    common = authorization["common_contract"]
    for key in ("fixed_split_seed", "fixed_model_seed", "cv_seed", "cv_enabled", "shuffle", "normalization_mode", "language_mode", "hash_verification_before_read", "fail_on_hash_mismatch", "output_root"):
        assert key in common and common[key] is not None
    defaults = authorization["execution_defaults"]
    assert defaults["tfidf_vectorizer"]["lowercase"] is False
    assert defaults["logistic_regression"]["solver"] == "lbfgs"
    assert defaults["linear_svm"]["random_state"] == 42
    assert defaults["multinomial_naive_bayes"]["fit_prior"] is True


def test_no_v1_split_policy_reference_in_active_repository_files():
    for directory in ("configs", "src", "scripts", "docs"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".py", ".md", ".txt", ".json"}:
                assert "nlp-split-policy-v1" not in path.read_text(encoding="utf-8", errors="replace").lower()
