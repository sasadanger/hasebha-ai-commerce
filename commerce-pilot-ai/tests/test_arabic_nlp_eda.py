"""Unit tests for the shared Arabic-NLP EDA helpers (src/nlp/eda_utils.py).

These cover the core, reusable building blocks used by
notebooks/02_arabic_nlp_eda_and_analysis.ipynb: length stats, script-ratio
detection, artifact detection, exact/near-duplicate detection, safe
normalization (original-preserving), and the TF-IDF baseline trainer.
"""
from __future__ import annotations

from src.nlp.eda_utils import (
    char_ngram_frequencies,
    detect_text_artifacts,
    find_exact_duplicates,
    find_near_duplicate_groups,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_length_histogram,
    safe_normalize_arabic,
    script_ratio,
    text_length_stats,
    train_tfidf_baseline,
    word_ngram_frequencies,
)


class TestTextLengthStats:
    def test_basic_counts(self):
        stats = text_length_stats(["hello world", "a", ""])
        assert stats["n"] == 3
        assert stats["chars"]["max"] == 11
        assert stats["chars"]["min"] == 0
        assert stats["words"]["max"] == 2
        assert stats["words"]["min"] == 0

    def test_handles_none_and_nan(self):
        stats = text_length_stats([None, float("nan"), "ok"])
        assert stats["n"] == 3
        assert stats["chars"]["min"] == 0

    def test_empty_iterable(self):
        stats = text_length_stats([])
        assert stats["n"] == 0
        assert stats["chars"]["mean"] == 0.0


class TestScriptRatio:
    def test_pure_arabic(self):
        result = script_ratio("مرحبا بكم في المتجر")
        assert result["arabic_ratio"] == 1.0
        assert result["latin_ratio"] == 0.0

    def test_pure_latin(self):
        result = script_ratio("hello world great product")
        assert result["latin_ratio"] == 1.0
        assert result["arabic_ratio"] == 0.0

    def test_mixed_arabic_and_latin(self):
        # Hand-written mixed string: "رائع" (4 Arabic letters) + "ok" (2 Latin letters).
        text = "رائع ok"
        result = script_ratio(text)
        assert result["n_chars"] == 6  # 4 Arabic + 2 Latin, whitespace excluded
        assert round(result["arabic_ratio"], 3) == round(4 / 6, 3)
        assert round(result["latin_ratio"], 3) == round(2 / 6, 3)
        assert result["other_ratio"] == 0.0

    def test_empty_text(self):
        result = script_ratio("")
        assert result == {"arabic_ratio": 0.0, "latin_ratio": 0.0, "other_ratio": 0.0, "n_chars": 0}

    def test_none_text(self):
        result = script_ratio(None)
        assert result["n_chars"] == 0


class TestDetectTextArtifacts:
    def test_detects_url_mention_hashtag(self):
        text = "check https://example.com @someuser #deal"
        result = detect_text_artifacts(text)
        assert result["has_url"] is True
        assert result["n_urls"] == 1
        assert result["has_mention"] is True
        assert result["has_hashtag"] is True

    def test_detects_diacritics_and_tatweel(self):
        text = "بِسْمِ اللَّهِ" + "قـــوي"
        result = detect_text_artifacts(text)
        assert result["has_diacritics"] is True
        assert result["has_tatweel"] is True

    def test_plain_text_has_no_artifacts(self):
        result = detect_text_artifacts("منتج جيد جدا")
        assert result["has_url"] is False
        assert result["has_mention"] is False
        assert result["has_diacritics"] is False
        assert result["has_tatweel"] is False


class TestNgramFrequencies:
    def test_word_unigrams(self):
        freqs = word_ngram_frequencies(["a b a", "a c"], n=1, top_k=5)
        counts = dict(freqs)
        assert counts["a"] == 3
        assert counts["b"] == 1

    def test_char_ngrams(self):
        freqs = char_ngram_frequencies(["aaa"], n=2, top_k=5)
        counts = dict(freqs)
        assert counts["aa"] == 2


class TestDuplicateDetection:
    def test_find_exact_duplicates_detects_repeats(self):
        texts = ["hello", "hello", "world", "  hello  "]
        result = find_exact_duplicates(texts)
        assert result["n_rows"] == 4
        # "hello" and "  hello  " normalize to the same key (whitespace collapse).
        assert result["n_duplicate_groups"] == 1
        assert result["n_rows_in_duplicate_groups"] == 3

    def test_find_exact_duplicates_no_dupes(self):
        result = find_exact_duplicates(["one", "two", "three"])
        assert result["n_duplicate_groups"] == 0
        assert result["n_rows_in_duplicate_groups"] == 0

    def test_find_near_duplicate_groups_catches_near_identical_text(self):
        texts = [
            "هذا المنتج رائع جدا وانصح به بشدة",
            "هذا المنتج رائع جدا وانصح به بشدة!!!",  # near-identical, extra punctuation
            "منتج سيء للغاية لا أنصح به أبدا",
        ]
        result = find_near_duplicate_groups(texts, jaccard_threshold=0.8)
        assert result["n_near_duplicate_pairs"] >= 1
        assert result["n_rows_involved"] >= 2

    def test_find_near_duplicate_groups_no_false_positive_on_distinct_text(self):
        texts = ["منتج ممتاز وسريع التوصيل", "لم يعجبني الحجم كان صغيرا جدا"]
        result = find_near_duplicate_groups(texts, jaccard_threshold=0.85)
        assert result["n_near_duplicate_pairs"] == 0


class TestSafeNormalizeArabic:
    def test_preserves_original_text(self):
        original = "  مرحباً @user https://x.co  إأآٱ !!!!!"
        result = safe_normalize_arabic(original)
        assert result["original"] == original  # untouched, byte-for-byte
        assert result["normalized"] != original
        assert result["changed"] is True

    def test_reports_unchanged_when_already_clean(self):
        clean = "منتج جيد"
        result = safe_normalize_arabic(clean)
        assert result["original"] == clean
        assert result["changed"] is False
        assert result["normalized"] == clean

    def test_none_input_does_not_raise(self):
        result = safe_normalize_arabic(None)
        assert result["original"] == ""
        assert result["normalized"] == ""


class TestTrainTfidfBaseline:
    def test_runs_end_to_end_and_returns_expected_shape(self):
        train_texts = [
            "منتج رائع جدا احببته",
            "سيء للغاية لا انصح به",
            "جودة عالية وسعر مناسب",
            "لم يعجبني الخامة رديئة",
            "توصيل سريع وخدمة ممتازة",
            "منتج مقلد ورديء الصنع",
        ]
        train_labels = ["POS", "NEG", "POS", "NEG", "POS", "NEG"]
        test_texts = ["منتج جيد ومناسب", "سيء ومخيب للامال"]
        test_labels = ["POS", "NEG"]

        result = train_tfidf_baseline(
            train_texts, train_labels, test_texts, test_labels,
            min_df=1, max_features=1000,
        )
        assert result.n_train == 6
        assert result.n_test == 2
        assert set(result.class_labels) == {"NEG", "POS"}
        assert 0.0 <= result.macro_f1 <= 1.0
        assert len(result.confusion_matrix) == 2
        assert set(result.per_class_f1.keys()) == {"NEG", "POS"}
        as_dict = result.to_dict()
        assert as_dict["n_train"] == 6


class TestPlottingHelpersReturnAxesWithLabels:
    def test_plot_class_distribution_sets_title_and_labels(self):
        ax = plot_class_distribution({"A": 3, "B": 7}, title="Test dist", xlabel="Cls", ylabel="N")
        assert ax.get_title() == "Test dist"
        assert ax.get_xlabel() == "Cls"
        assert ax.get_ylabel() == "N"

    def test_plot_length_histogram_sets_title(self):
        ax = plot_length_histogram([1, 2, 3, 4, 5], title="Lengths", bins=5)
        assert ax.get_title() == "Lengths"

    def test_plot_confusion_matrix_sets_ticks(self):
        ax = plot_confusion_matrix([[5, 1], [2, 4]], ["A", "B"], title="CM")
        assert ax.get_title() == "CM"
        assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
