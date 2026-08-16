import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.nlp.duplicate_control import normalized_exact_key, raw_exact_key
from src.nlp.text_normalization import is_empty_text, normalize_text

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def reproduce(dataset):
    return json.loads(subprocess.check_output(
        [sys.executable, "scripts/reaudit_nlp_duplicates.py", dataset],
        cwd=ROOT, encoding="utf-8"
    ))


def test_latin_only_lowercase_english():
    assert normalize_text("AbC XYZ") == "abc xyz"


def test_latin_only_lowercase_does_not_lowercase_non_latin_script():
    assert normalize_text("ΑΒΓ БВГ") == "ΑΒΓ БВГ"


def test_arabic_and_mixed_text_preserved_except_contract_arabic_rules():
    assert normalize_text("مرحبا HELLO 123 😀") == "مرحبا hello 123 😀"


def test_hashtag_url_mention_and_punctuation_rules():
    assert normalize_text("#Topic HTTPS://EXAMPLE.COM @User !!!!") == "topic [URL] [MENTION] [REPEAT_PUNCT]"


def test_repeated_punctuation_normalizes_but_symbols_and_emoji_are_preserved():
    for value in ("!!!", "???", "،،،", "---", "___", "………", "......"):
        assert normalize_text(value) == "[REPEAT_PUNCT]"
    assert normalize_text("……") == "……"
    for value in ("😀😀😀", "❤️❤️❤️", "⭐⭐⭐", "🔥🔥🔥", "★★★", "©©©"):
        assert normalize_text(value) == value


def test_normalization_is_deterministic():
    sample = "  #MiXeD\u200b!!!  "
    assert normalize_text(sample) == normalize_text(sample)


def test_empty_text_minimal_stage_only():
    assert is_empty_text(None)
    assert is_empty_text(" \t\n\u200b\ufeff ")
    for value in ("ـ", "َ", "!", "😀", "https://x.test", "@user", "#tag", "123"):
        assert not is_empty_text(value)


def test_empty_semantics_are_separate_from_full_normalization():
    assert normalize_text("ـ") == ""
    assert not is_empty_text("ـ")


def test_duplicate_keys_are_exact_and_deterministic():
    assert raw_exact_key(" A ") != raw_exact_key("a")
    assert normalized_exact_key(" A ") == normalized_exact_key("a")


def test_v2_contract_pins_complete_pipeline():
    c = load_yaml("configs/nlp_text_normalization_contract_v2.yaml")
    steps = " ".join(c["normalized_duplicate_stage"]["ordered_steps"])
    assert "hashtag" not in steps.lower() or "Strip #" in steps
    assert "3+" in steps and "LATIN" in steps


def test_egyptian_reaudit_v2_exact():
    d = reproduce("egyptian_tweets_40k")
    assert (d["actual_row_count"], d["raw_exact_duplicate_rows"], d["normalized_exact_duplicate_rows"]) == (40000, 323, 595)
    assert d["same_normalized_text_same_combined_label_duplicate_rows"] == 580
    assert d["per_label_conflicts"]["label"] == {"key_count": 15, "row_count": 31}
    assert (d["unique_normalized_text_count"], d["missing_or_empty_text"]) == (39405, 0)


def test_arsas_reaudit_v2_exact():
    d = reproduce("arsas")
    assert (d["actual_row_count"], d["raw_exact_duplicate_rows"], d["normalized_exact_duplicate_rows"]) == (19897, 99, 121)
    assert d["per_label_conflicts"]["Sentiment_label"] == {"key_count": 31, "row_count": 62}
    assert d["per_label_conflicts"]["Speech_act_label"] == {"key_count": 16, "row_count": 32}
    assert (d["unique_normalized_text_count"], d["missing_or_empty_text"]) == (19776, 0)


def test_amazon_and_labr_targets_are_explicit_native_ratings():
    m = load_yaml("configs/nlp_metric_contract_v2.yaml")["batch1_experiments"]
    for key in ("EXPERIMENT_A_english_ecommerce_review_baseline", "EXPERIMENT_C_arabic_review_domain_robustness"):
        assert m[key]["task_type"] == "FIVE_CLASS_RATING_CLASSIFICATION"
        assert m[key]["labels"] == [1, 2, 3, 4, 5]
        assert m[key]["derived_mapping"] == "NONE"
    assert m["EXPERIMENT_A_english_ecommerce_review_baseline"]["source_label"] == "overall"


def test_every_batch1_metric_is_defined_and_no_binned_metric_exists():
    m = load_yaml("configs/nlp_metric_contract_v2.yaml")
    assert len(m["batch1_experiments"]) == 4
    assert all(x["primary_metric"] == "macro_f1" for x in m["batch1_experiments"].values())
    assert "binned" not in (ROOT / "configs/nlp_metric_contract_v2.yaml").read_text(encoding="utf-8").lower()


def test_learned_configuration_counts_are_exact_and_bounded():
    a = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")
    counts = {x["experiment_id"]: len(x["learned_configurations"]) for x in a["BATCH_1_CANDIDATE"]}
    assert counts == {"A": 4, "B2": 6, "C": 4, "E": 6}
    assert max(counts.values()) <= a["common_contract"]["maximum_learned_configurations_per_experiment"] == 6


def test_no_search_transformer_embedding_or_neural_authorization():
    b = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")["future_candidate_boundaries"]
    assert not any(b[k] for k in b if k.endswith("authorized") and k != "effective_only_after_independent_approval")


def test_batch1_contains_only_expected_ready_active_experiments():
    a = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")
    assert [(x["experiment_id"], x["status"]) for x in a["BATCH_1_CANDIDATE"]] == [("A", "READY_FOR_REVIEW"), ("B2", "READY_FOR_REVIEW"), ("C", "READY_FOR_REVIEW"), ("E", "READY_FOR_REVIEW")]


def test_batch2_is_preserved():
    a = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")
    assert [x["experiment_id"] for x in a["BATCH_2_PRESERVED"]] == ["B1", "B3", "D1"]


def test_d2_f_g_h_remain_blocked():
    a = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")
    assert [x["experiment_id"] for x in a["BLOCKED_FUTURE"]] == ["D2", "F", "G", "H"]
    assert all(x["status"].startswith("BLOCKED") for x in a["BLOCKED_FUTURE"])


def test_current_session_authorizes_no_execution():
    c = load_yaml("configs/nlp_training_batch_authorization_v2.yaml")["current_session"]
    assert c["training_executed"] is False
    assert not any(v for k, v in c.items() if k.endswith("authorized"))


def test_no_protected_test_reference_in_executable_nlp_paths():
    for path in ("src/nlp/text_normalization.py", "src/nlp/duplicate_control.py", "scripts/reaudit_nlp_duplicates.py"):
        text = (ROOT / path).read_text(encoding="utf-8").lower()
        assert "phase2a" not in text and "test_catboost" not in text
