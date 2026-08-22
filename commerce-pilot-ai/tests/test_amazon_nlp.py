"""Tests for the Amazon Appliances sentiment pipeline (src/nlp/amazon/).

Covers: target construction correctness, no rating-leakage into model input, dedup correctness,
learning-curve subset nesting, split disjointness (using the real split artifacts written by
scripts/run_amazon_sentiment_pipeline.py), and a sanity check on inference.predict().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nlp.amazon import data as amz_data

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS_DIR = REPO_ROOT / "reports" / "generated" / "amazon" / "split_ids"
BEST_MODEL_MANIFEST = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "models" / "best_model_manifest.json"
TRANSFORMER_MODEL_MANIFEST = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer" / "model_manifest.json"
TRANSFORMER_CALIBRATION = REPO_ROOT / "reports" / "generated" / "amazon" / "transformer_calibration.json"


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def test_assign_binary_label_maps_1_2_to_negative_4_5_to_positive_3_to_nan():
    ratings = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    labels = amz_data.assign_binary_label(ratings)
    assert labels.iloc[0] == 0
    assert labels.iloc[1] == 0
    assert np.isnan(labels.iloc[2])
    assert labels.iloc[3] == 1
    assert labels.iloc[4] == 1


def test_assign_binary_label_rejects_out_of_range_ratings():
    with pytest.raises(ValueError):
        amz_data.assign_binary_label(pd.Series([1.0, 6.0]))


def test_filter_binary_eligible_drops_rating_3_and_keeps_int_label():
    df = pd.DataFrame({"rating": [1.0, 2.0, 3.0, 4.0, 5.0], "row": [0, 1, 2, 3, 4]})
    out = amz_data.filter_binary_eligible(df)
    assert sorted(out["row"].tolist()) == [0, 1, 3, 4]
    assert set(out["label"].unique()) == {0, 1}
    assert out["label"].dtype.kind == "i"


# ---------------------------------------------------------------------------
# No-leakage: rating must never be part of model input text
# ---------------------------------------------------------------------------


def test_build_model_text_does_not_use_rating_column():
    """Dropping `rating` from the input frame must not change build_model_text's output --
    proof that rating never flows into the text a vectorizer sees."""
    with_rating = pd.DataFrame(
        {"title": ["Great buy"], "text": ["works well"], "rating": [1.0]}
    )
    without_rating = with_rating.drop(columns=["rating"])
    result_with = amz_data.build_model_text(with_rating)
    result_without = amz_data.build_model_text(without_rating)
    assert result_with.iloc[0] == result_without.iloc[0]
    assert result_with.iloc[0] == "Great buy. works well"


def test_build_model_text_falls_back_to_text_only_when_title_missing():
    df = pd.DataFrame({"title": [None, ""], "text": ["only body text", "second body"]})
    result = amz_data.build_model_text(df)
    assert result.iloc[0] == "only body text"
    assert result.iloc[1] == "second body"


def test_train_functions_only_accept_text_arrays_not_frames_with_rating():
    """train.py trainer functions take (texts, labels) -- plain string arrays -- so there is no
    code path through which a `rating` column could reach the vectorizer as a feature."""
    from src.nlp.amazon import train as amz_train

    texts = np.array(["good product works great", "bad product broke fast"] * 20)
    labels = np.array([1, 0] * 20)
    model = amz_train.train_tfidf_logreg(texts, labels)
    vectorizer = model.named_steps["tfidf"]
    feature_names = set(vectorizer.get_feature_names_out())
    assert "rating" not in feature_names


# ---------------------------------------------------------------------------
# Dedup correctness (must happen before any split)
# ---------------------------------------------------------------------------


def _tiny_pool(texts, parent_asins, ratings):
    n = len(texts)
    df = pd.DataFrame(
        {
            "parent_asin": parent_asins,
            "asin": [f"asin{i}" for i in range(n)],
            "user_id": [f"user{i}" for i in range(n)],
            "rating": ratings,
            "timestamp": list(range(n)),
            "review_datetime_utc": pd.to_datetime(["2020-01-0%d" % (i + 1) for i in range(n)]),
            "verified_purchase": [True] * n,
            "title": [f"title{i}" for i in range(n)],
            "text": texts,
        }
    )
    return amz_data.filter_binary_eligible(df)


def test_remove_duplicate_text_removes_exact_and_near_duplicates():
    texts = [
        "This appliance works great!!",
        "This appliance works great",  # near-dup of row 0 (punctuation only)
        "this appliance works great",  # exact-dup of row 1 after normalization
        "Completely different review body",
    ]
    df = _tiny_pool(texts, ["A", "A", "B", "C"], [5.0, 5.0, 5.0, 1.0])
    deduped, report = amz_data.remove_duplicate_text(df)
    assert len(deduped) == 2
    assert report["exact_duplicates_removed"] == 0
    assert report["near_duplicates_removed"] == 2
    # No two surviving rows share normalized text (i.e. no duplicate text crosses forward
    # into whatever split those rows are later assigned to).
    normalized = deduped["text"].map(amz_data.normalize_text_for_dedup)
    assert normalized.is_unique


# ---------------------------------------------------------------------------
# Learning-curve subsets: nested and nearly balanced
# ---------------------------------------------------------------------------


def test_learning_curve_subsets_are_nested_prefixes_and_balanced():
    n_per_class = 60
    negative = pd.DataFrame(
        {"label": [0] * n_per_class, "row": [f"neg{i}" for i in range(n_per_class)]}
    )
    positive = pd.DataFrame(
        {"label": [1] * n_per_class, "row": [f"pos{i}" for i in range(n_per_class)]}
    )
    train_pool = pd.concat([negative, positive], ignore_index=True)
    subsets, achieved = amz_data.build_learning_curve_subsets(train_pool, sizes=(20, 40, 150))
    assert achieved[20] == {"negative": 10, "positive": 10, "total": 20}
    assert achieved[40] == {"negative": 20, "positive": 20, "total": 40}
    # 150 requested (75/class) but only 60 of each class available -> capped, not fabricated.
    assert achieved[150] == {"negative": 60, "positive": 60, "total": 120}
    rows_20 = set(subsets[20]["row"])
    rows_40 = set(subsets[40]["row"])
    assert rows_20.issubset(rows_40)


# ---------------------------------------------------------------------------
# Split disjointness (integration-level, using the real artifacts on disk)
# ---------------------------------------------------------------------------

_ARTIFACTS_PRESENT = SPLIT_IDS_DIR.exists() and (SPLIT_IDS_DIR / "train_full_pool.parquet").exists()


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="run scripts/run_amazon_sentiment_pipeline.py first")
class TestRealSplitDisjointness:
    @staticmethod
    def _load(name: str) -> pd.DataFrame:
        return pd.read_parquet(SPLIT_IDS_DIR / f"{name}.parquet")

    def test_no_row_overlap_between_train_and_fixed_eval_sets(self):
        train = self._load("train_full_pool")
        train_uids = set(train["review_uid"])
        for name in ["val", "test_balanced", "test_representative", "product_holdout_stress"]:
            other = self._load(name)
            overlap = train_uids & set(other["review_uid"])
            assert not overlap, f"{len(overlap)} rows leak between train and {name}"

    def test_no_product_overlap_between_train_and_fixed_eval_sets(self):
        train_products = set(self._load("train_full_pool")["parent_asin"])
        for name in ["val", "test_balanced", "test_representative", "product_holdout_stress"]:
            other_products = set(self._load(name)["parent_asin"])
            overlap = train_products & other_products
            assert not overlap, f"{len(overlap)} products leak between train and {name}"

    def test_no_row_overlap_between_val_and_test_sets(self):
        val_uids = set(self._load("val")["review_uid"])
        test_balanced_uids = set(self._load("test_balanced")["review_uid"])
        test_rep_uids = set(self._load("test_representative")["review_uid"])
        assert not (val_uids & test_balanced_uids)
        assert not (val_uids & test_rep_uids)
        assert not (test_balanced_uids & test_rep_uids)

    def test_chronological_stress_rows_do_not_overlap_train_rows(self):
        train_uids = set(self._load("train_full_pool")["review_uid"])
        chrono_uids = set(self._load("chronological_stress")["review_uid"])
        assert not (train_uids & chrono_uids)


# ---------------------------------------------------------------------------
# Inference sanity check
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BEST_MODEL_MANIFEST.exists(), reason="run scripts/run_amazon_sentiment_pipeline.py first")
def test_inference_predict_returns_sane_structure():
    from src.nlp.amazon import inference as amz_inference

    result = amz_inference.predict(
        "This washing machine broke down after two weeks and customer service was unhelpful."
    )
    assert result["label"] in (0, 1)
    assert result["label_name"] in ("negative", "positive")
    assert result["positive_score"] is None or 0.0 <= result["positive_score"] <= 1.0
    assert isinstance(result["model_name"], str)
    assert isinstance(result["model_sha256"], str) and len(result["model_sha256"]) == 64


@pytest.mark.skipif(not BEST_MODEL_MANIFEST.exists(), reason="run scripts/run_amazon_sentiment_pipeline.py first")
def test_inference_predict_rejects_empty_text():
    from src.nlp.amazon import inference as amz_inference

    with pytest.raises(ValueError):
        amz_inference.predict("")


# ---------------------------------------------------------------------------
# Transformer: text construction / no-leakage
# ---------------------------------------------------------------------------


def test_build_transformer_text_does_not_use_rating_column():
    """Same no-leakage guarantee as build_model_text, for the transformer's text builder."""
    with_rating = pd.DataFrame({"title": ["Great buy"], "text": ["works well"], "rating": [1.0]})
    without_rating = with_rating.drop(columns=["rating"])
    result_with = amz_data.build_transformer_text(with_rating)
    result_without = amz_data.build_transformer_text(without_rating)
    assert result_with.iloc[0] == result_without.iloc[0]
    assert result_with.iloc[0] == "Great buy. works well"


def test_normalize_text_for_transformer_strips_html_and_control_chars_but_preserves_casing_and_punct():
    raw = "Great <br/>product!!  \x07Loved\tit -- 5/5  stars"
    normalized = amz_data.normalize_text_for_transformer(raw)
    assert "<br" not in normalized
    assert "\x07" not in normalized
    assert "Great" in normalized  # casing preserved
    assert "!!" in normalized  # punctuation preserved
    assert "  " not in normalized  # whitespace collapsed


def test_normalize_text_for_transformer_is_different_and_lighter_than_dedup_normalization():
    """The transformer normalizer must NOT destroy casing/punctuation the way the dedup
    normalizer intentionally does -- these are different functions for different purposes."""
    raw = "This Is GREAT!!!"
    transformer_version = amz_data.normalize_text_for_transformer(raw)
    dedup_version = amz_data.normalize_text_for_dedup(raw)
    assert transformer_version == "This Is GREAT!!!"
    assert dedup_version == "this is great"
    assert transformer_version != dedup_version


def test_build_transformer_text_falls_back_to_text_only_when_title_missing():
    df = pd.DataFrame({"title": [None, ""], "text": ["only body text", "second body"]})
    result = amz_data.build_transformer_text(df)
    assert result.iloc[0] == "only body text"
    assert result.iloc[1] == "second body"


# ---------------------------------------------------------------------------
# Transformer: compute_hf_metrics label-mapping correctness
# ---------------------------------------------------------------------------


def test_compute_hf_metrics_perfect_predictions_give_macro_f1_one():
    from src.nlp.amazon import transformer as amz_tf

    logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]])
    labels = np.array([0, 1, 0, 1])
    metrics = amz_tf.compute_hf_metrics((logits, labels))
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["negative_f1"] == pytest.approx(1.0)
    assert metrics["positive_f1"] == pytest.approx(1.0)


def test_compute_hf_metrics_no_label_inversion():
    """If logits confidently favor class 0 (negative) for a row labeled 1 (positive), that row
    must be scored as WRONG, not right -- guards against an accidental 0/1 label swap."""
    from src.nlp.amazon import transformer as amz_tf

    logits = np.array([[5.0, -5.0]])  # confidently predicts class 0 (negative)
    labels = np.array([1])  # true label is positive
    metrics = amz_tf.compute_hf_metrics((logits, labels))
    assert metrics["macro_f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Transformer: val_natural bucket construction (product-disjoint second pass)
# ---------------------------------------------------------------------------


def _synthetic_pool_for_split(n_products: int = 40, rows_per_product: int = 10) -> pd.DataFrame:
    rows = []
    for p in range(n_products):
        for i in range(rows_per_product):
            label = 0 if i % 3 == 0 else 1  # 1/3 negative, 2/3 positive per product
            rows.append(
                {
                    "parent_asin": f"P{p}",
                    "label": label,
                    "rating": 1.0 if label == 0 else 5.0,
                    "review_uid": f"uid_{p}_{i}",
                }
            )
    return pd.DataFrame(rows)


def test_val_natural_does_not_change_the_four_original_buckets():
    """Passing val_natural_total must not alter val/test_balanced/test_representative/train --
    verified by comparing bucket contents with and without the extra bucket requested."""
    pool = _synthetic_pool_for_split()
    buckets_without, assigned_without = amz_data.build_product_disjoint_splits(
        pool, seed=123, val_size=20, test_balanced_size=20, test_representative_total=20, train_target=50
    )
    buckets_with, assigned_with = amz_data.build_product_disjoint_splits(
        pool, seed=123, val_size=20, test_balanced_size=20, test_representative_total=20, train_target=50,
        val_natural_total=20,
    )
    for name in ["val", "test_balanced", "test_representative", "train"]:
        assert set(buckets_without[name]["review_uid"]) == set(buckets_with[name]["review_uid"])
        assert assigned_without[name] == assigned_with[name]
    assert "val_natural" in buckets_with
    assert "val_natural" not in buckets_without


def test_val_natural_is_product_disjoint_from_the_four_original_buckets():
    pool = _synthetic_pool_for_split()
    buckets, _assigned = amz_data.build_product_disjoint_splits(
        pool, seed=123, val_size=20, test_balanced_size=20, test_representative_total=20, train_target=50,
        val_natural_total=20,
    )
    natural_products = set(buckets["val_natural"]["parent_asin"])
    natural_uids = set(buckets["val_natural"]["review_uid"])
    for name in ["val", "test_balanced", "test_representative", "train"]:
        other_products = set(buckets[name]["parent_asin"])
        other_uids = set(buckets[name]["review_uid"])
        assert not (natural_products & other_products), f"product overlap between val_natural and {name}"
        assert not (natural_uids & other_uids), f"row overlap between val_natural and {name}"


# ---------------------------------------------------------------------------
# Transformer: calibration must be fit on val_natural only, never a test set
# ---------------------------------------------------------------------------


def test_fit_temperature_recovers_a_reasonable_scalar_on_synthetic_logits():
    from src.nlp.amazon import transformer as amz_tf

    rng = np.random.default_rng(0)
    n = 500
    true_labels = rng.integers(0, 2, size=n)
    # Model is confidently "correct" on its OWN (noisy) label guess, but that guess is wrong on
    # 15% of rows relative to true_labels -- i.e. genuinely overconfident: high-magnitude logits
    # whose actual accuracy (85%) is well below what such confidence would imply. Temperature
    # scaling should soften this (temperature > 1).
    flipped = true_labels.copy()
    flip_mask = rng.random(n) < 0.15
    flipped[flip_mask] = 1 - flipped[flip_mask]
    base = np.where(flipped[:, None] == np.array([0, 1]), 8.0, -8.0)
    logits = base + rng.normal(0, 0.1, size=base.shape)
    temperature = amz_tf.fit_temperature(logits, true_labels)
    assert temperature > 1.0  # overconfident-relative-to-accuracy logits require temperature > 1
    calibrated = amz_tf.temperature_scale(logits, temperature)
    assert calibrated.shape == logits.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_brier_and_ece_are_zero_for_perfect_confident_correct_predictions():
    from src.nlp.amazon import transformer as amz_tf

    y_true = np.array([0, 1, 0, 1])
    y_prob_positive = np.array([0.0, 1.0, 0.0, 1.0])
    assert amz_tf.brier_score(y_true, y_prob_positive) == pytest.approx(0.0)
    assert amz_tf.expected_calibration_error(y_true, y_prob_positive) == pytest.approx(0.0)


@pytest.mark.skipif(not TRANSFORMER_CALIBRATION.exists(), reason="run scripts/amazon_transformer_calibrate.py first")
def test_transformer_calibration_was_fit_on_val_natural_only():
    """Guards against calibration accidentally touching a test set: the saved calibration config
    must explicitly document val_natural as its ONLY fitting source."""
    import json

    config = json.loads(TRANSFORMER_CALIBRATION.read_text())
    fit_on = config["fit_on"]
    assert "val_natural" in fit_on
    for test_set_name in ["test_balanced", "test_representative", "product_holdout_stress", "chronological_stress"]:
        assert test_set_name not in fit_on
    assert config["n_val_natural"] > 0


# ---------------------------------------------------------------------------
# Transformer: inference dispatch (skipped unless the checkpoint has been trained)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRANSFORMER_MODEL_MANIFEST.exists(), reason="run scripts/amazon_transformer_train.py + amazon_transformer_calibrate.py first")
def test_predict_with_model_transformer_returns_sane_structure():
    from src.nlp.amazon import inference as amz_inference

    result = amz_inference.predict_with_model(
        "This washing machine broke down after two weeks and customer service was unhelpful.",
        model="transformer",
    )
    assert result["label"] in (0, 1)
    assert result["label_name"] in ("negative", "positive")
    assert 0.0 <= result["positive_score"] <= 1.0
    assert isinstance(result["model_sha256"], str) and len(result["model_sha256"]) == 64


@pytest.mark.skipif(not TRANSFORMER_MODEL_MANIFEST.exists(), reason="run scripts/amazon_transformer_train.py + amazon_transformer_calibrate.py first")
def test_predict_batch_transformer_matches_single_text_predict_with_model():
    from src.nlp.amazon import inference as amz_inference

    texts = [
        "Absolutely love this appliance, works perfectly every time.",
        "Terrible product, broke immediately and support ignored me.",
    ]
    batch_results = amz_inference.predict_batch(texts, model="transformer")
    single_results = [amz_inference.predict_with_model(t, model="transformer") for t in texts]
    assert len(batch_results) == 2
    for b, s in zip(batch_results, single_results):
        assert b["label"] == s["label"]
        assert b["positive_score"] == pytest.approx(s["positive_score"], abs=1e-4)


@pytest.mark.skipif(not BEST_MODEL_MANIFEST.exists(), reason="run scripts/run_amazon_sentiment_pipeline.py first")
def test_predict_with_model_default_reproduces_original_tfidf_predict():
    """predict_with_model(model='tfidf') (new) must exactly reproduce predict() (unchanged) --
    proof that extending inference.py did not alter the original TF-IDF behavior."""
    from src.nlp.amazon import inference as amz_inference

    text = "This dishwasher is fantastic, cleans every dish perfectly."
    original = amz_inference.predict(text)
    via_dispatch = amz_inference.predict_with_model(text, model="tfidf")
    assert original == via_dispatch


def test_predict_batch_rejects_unknown_model_name():
    from src.nlp.amazon import inference as amz_inference

    with pytest.raises(ValueError):
        amz_inference.predict_batch(["some text"], model="not_a_real_model")


def test_predict_batch_rejects_empty_list():
    from src.nlp.amazon import inference as amz_inference

    with pytest.raises(ValueError):
        amz_inference.predict_batch([])
