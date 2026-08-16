import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.nlp.amazon_adapter import CANONICAL_FIELDS, adapt_amazon_record
from src.nlp.configuration import instantiate_configuration, resolve_batch1_configurations
from src.nlp.execution_control import (
    InternalTestAccessDenied,
    PartitionStore,
    ReleaseAuthorization,
    fit_model_train_only,
    fit_train_only_transform,
    select_winner,
)
from src.nlp.splitting import materialize_group_split, materialize_text_group_split
from src.nlp.text_normalization import normalize_text

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "configs/nlp_training_batch_authorization_v2.yaml"


def test_real_amazon_physical_fixture_maps_to_canonical_schema():
    physical = {"rating": 4.0, "title": "Good", "text": "Works", "asin": "x"}
    assert adapt_amazon_record(physical) == {
        "overall": 4.0, "review_title": "Good", "review_text": "Works"
    }
    assert CANONICAL_FIELDS == ("overall", "review_title", "review_text")


@pytest.mark.parametrize("missing", ["rating", "title", "text"])
def test_amazon_adapter_fails_when_physical_field_is_missing(missing):
    physical = {"rating": 4.0, "title": "Good", "text": "Works"}
    del physical[missing]
    with pytest.raises(ValueError, match="missing required fields"):
        adapt_amazon_record(physical)


def test_amazon_adapter_rejects_ambiguous_precanonicalized_input():
    with pytest.raises(ValueError, match="unexpectedly contains canonical"):
        adapt_amazon_record({"rating": 4, "title": "x", "text": "y", "overall": 4})


def test_unicode_punctuation_contract_exact():
    for value in ("!!!", "???", "،،،", "---", "___", "………"):
        assert normalize_text(value) == "[REPEAT_PUNCT]"
    assert normalize_text("……") == "……"
    for value in ("😀😀😀", "🔥🔥", "❤️❤️", "★★★", "✓✓✓", "→→→", "©©", "™™", "aaa", "111"):
        assert normalize_text(value) == value
    assert normalize_text("good!!!😀😀") == "good[REPEAT_PUNCT]😀😀"
    assert normalize_text("wow???★★") == "wow[REPEAT_PUNCT]★★"
    assert normalize_text("hello………🔥🔥") == "hello[REPEAT_PUNCT]🔥🔥"


def test_active_dependency_chain_is_v2_only():
    manifest = yaml.safe_load((ROOT / "configs/nlp_experiment_manifest.yaml").read_text(encoding="utf-8"))
    split = yaml.safe_load((ROOT / manifest["active_split_policy"]).read_text(encoding="utf-8"))
    duplicate_path = split["general_split_requirements"]["duplicate_contract"]
    duplicate = yaml.safe_load((ROOT / duplicate_path).read_text(encoding="utf-8"))
    assert manifest["active_split_policy_version"] == "nlp-split-policy-v2"
    assert duplicate["schema_version"] == "nlp-duplicate-control-contract-v2"
    active_blob = json.dumps({"manifest": manifest, "split": split}).lower()
    assert "nlp-split-policy-v1" not in active_blob
    assert "configs/nlp_duplicate_control_contract.yaml" not in active_blob


def synthetic_records(per_class=100):
    return [
        {"group": f"{label}-{index:03d}", "label": label, "text": f"row-{label}-{index}"}
        for label in ("a", "b") for index in range(per_class)
    ]


def test_group_split_is_deterministic_isolated_and_proportional():
    rows = synthetic_records()
    first = materialize_group_split(rows, group_key="group", label_key="label", seed=20260809)
    second = materialize_group_split(rows, group_key="group", label_key="label", seed=20260809)
    assert first == second
    assignments = {x.group_key: x.split for x in first}
    assert len(assignments) == 200
    counts = Counter(assignments.values())
    assert counts == {"train": 140, "validation": 30, "internal_test": 30}


def test_group_split_keeps_duplicate_rows_together_and_stratifies():
    rows = synthetic_records()
    rows.extend([{"group": "a-000", "label": "a", "text": "duplicate"} for _ in range(3)])
    result = materialize_group_split(rows, group_key="group", label_key="label", seed=20260809)
    group_splits = {}
    for item in result:
        group_splits.setdefault(item.group_key, set()).add(item.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    index_to_label = {index: row["label"] for index, row in enumerate(rows)}
    distributions = Counter((item.split, index_to_label[item.row_index]) for item in result)
    assert distributions[("train", "a")] == distributions[("train", "b")] + 3
    assert distributions[("validation", "a")] == distributions[("validation", "b")]
    assert distributions[("internal_test", "a")] == distributions[("internal_test", "b")]


def test_text_group_split_constructs_normalized_duplicate_keys():
    rows = [
        {"text": f"unique {label} {index}", "label": label}
        for label in ("a", "b") for index in range(40)
    ]
    rows.extend([{"text": " DUPLICATE!!! ", "label": "a"}, {"text": "duplicate???", "label": "a"}])
    # The two punctuation forms normalize to the same key and must stay together.
    assignments = materialize_text_group_split(rows, text_key="text", label_key="label", seed=9)
    by_index = {item.row_index: item.split for item in assignments}
    assert by_index[len(rows) - 2] == by_index[len(rows) - 1]


def test_group_split_invalid_and_conflicting_inputs_fail_closed():
    with pytest.raises(ValueError):
        materialize_group_split([], group_key="group", label_key="label", seed=1)
    rows = synthetic_records(3) + [{"group": "a-000", "label": "b"}]
    with pytest.raises(ValueError, match="conflicting labels"):
        materialize_group_split(rows, group_key="group", label_key="label", seed=1)


class SpyTransformer:
    def __init__(self):
        self.fit_rows = None
        self.transformed = []

    def fit_transform(self, rows):
        self.fit_rows = list(rows)
        return [f"fit:{x}" for x in rows]

    def transform(self, rows):
        self.transformed.append(list(rows))
        return [f"transform:{x}" for x in rows]


def test_preprocessing_and_tfidf_boundary_fits_train_only():
    store = PartitionStore(["TRAIN_SENTINEL"], ["VALIDATION_SENTINEL"], ["INTERNAL_TEST_SENTINEL"])
    spy = SpyTransformer()
    train_features, validation_features = fit_train_only_transform(store, spy)
    assert spy.fit_rows == ["TRAIN_SENTINEL"]
    assert spy.transformed == [["VALIDATION_SENTINEL"]]
    assert "INTERNAL_TEST_SENTINEL" not in repr((spy.fit_rows, spy.transformed))
    assert train_features == ["fit:TRAIN_SENTINEL"]
    assert validation_features == ["transform:VALIDATION_SENTINEL"]


class SpyEstimator:
    def __init__(self):
        self.fit_payload = None

    def fit(self, features, labels):
        self.fit_payload = (list(features), list(labels))
        return self


def test_model_fit_api_accepts_only_explicit_train_payload():
    estimator = SpyEstimator()
    fit_model_train_only(estimator, ["TRAIN_FEATURE"], ["TRAIN_LABEL"])
    assert estimator.fit_payload == (["TRAIN_FEATURE"], ["TRAIN_LABEL"])
    with pytest.raises(ValueError, match="length mismatch"):
        fit_model_train_only(estimator, [1, 2], [1])


def authorization(authorized=True):
    return ReleaseAuthorization(
        authorized=authorized, run_id="synthetic-run", winning_compound_id="A::cfg",
        winning_configuration_fingerprint="a" * 64, reason="synthetic gate test",
        metric_contract="nlp-metric-contract-v2", split_policy="nlp-split-policy-v2",
    )


def test_internal_test_denied_before_release_and_release_is_audited(tmp_path):
    store = PartitionStore(["train"], ["validation"], ["internal"])
    with pytest.raises(InternalTestAccessDenied):
        store.internal_test()
    with pytest.raises(InternalTestAccessDenied):
        store.evaluate_internal_test_once(authorization(False), tmp_path / "ledger.json", lambda rows: len(rows))
    ledger = tmp_path / "ledger.json"
    assert store.evaluate_internal_test_once(authorization(), ledger, lambda rows: {"evaluated_rows": len(rows)}) == {"evaluated_rows": 1}
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["release_state"] == "RELEASED_ONCE"
    assert event["winning_configuration_fingerprint"] == "a" * 64
    with pytest.raises(InternalTestAccessDenied):
        store.evaluate_internal_test_once(authorization(), ledger, lambda rows: len(rows))


def test_all_20_configs_resolve_in_order_and_instantiate_without_fit():
    configurations = resolve_batch1_configurations(AUTH)
    assert Counter(x["experiment_id"] for x in configurations) == {"A": 4, "B2": 6, "C": 4, "E": 6}
    assert len(configurations) == len({x["compound_id"] for x in configurations}) == 20
    assert [x["execution_order"] for x in configurations] == list(range(20))
    assert [x["experiment_id"] for x in configurations] == ["A"] * 4 + ["B2"] * 6 + ["C"] * 4 + ["E"] * 6
    for configuration in configurations:
        vectorizer, estimator = instantiate_configuration(configuration)
        assert vectorizer is not None and estimator is not None


def test_configuration_resolution_and_fingerprints_are_deterministic():
    first = resolve_batch1_configurations(AUTH)
    second = resolve_batch1_configurations(AUTH)
    assert first == second
    assert len({x["fingerprint_sha256"] for x in first}) == 20


def test_configuration_override_precedence():
    configurations = resolve_batch1_configurations(AUTH)
    unigram = next(x for x in configurations if x["compound_id"] == "A::tfidf_word_unigram_logreg")
    assert unigram["vectorizer"]["min_df"] == 2
    assert unigram["vectorizer"]["max_df"] == 1.0
    assert unigram["classifier"]["max_iter"] == 1000


def test_unknown_configuration_key_is_rejected(tmp_path):
    data = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    data["BATCH_1_CANDIDATE"][0]["learned_configurations"][0]["surprise"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration keys"):
        resolve_batch1_configurations(path)


def test_unknown_nested_parameter_is_rejected(tmp_path):
    data = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    data["execution_defaults"]["tfidf_vectorizer"]["unknown_vectorizer_knob"] = True
    path = tmp_path / "invalid-nested.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown executable parameters"):
        resolve_batch1_configurations(path)


def test_winner_selection_and_complete_tie_breaking():
    base = {"experiment_id": "A", "macro_f1": 0.8, "balanced_accuracy": 0.7, "accuracy": 0.9}
    results = [
        {**base, "compound_id": "late", "execution_order": 2},
        {**base, "compound_id": "early", "execution_order": 1},
        {**base, "compound_id": "lower-balanced", "balanced_accuracy": 0.6, "execution_order": 0},
    ]
    assert select_winner(results)["compound_id"] == "early"
    with pytest.raises(ValueError, match="different experiments"):
        select_winner([results[0], {**results[1], "experiment_id": "B2"}])


def test_optional_stages_explicitly_disabled_and_no_transformer_candidate():
    authorization_data = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    common = authorization_data["common_contract"]
    disabled = [key for key in common if key.endswith("_enabled") and key != "cv_enabled"]
    assert disabled and all(common[key] is False for key in disabled)
    manifest_text = (ROOT / "configs/nlp_experiment_manifest.yaml").read_text(encoding="utf-8").lower()
    assert not any(term in manifest_text for term in ("transformer", "embedding", "bert", "roberta", "xlm"))
