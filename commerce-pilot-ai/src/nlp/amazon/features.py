"""TF-IDF vectorizer construction for Amazon Appliances sentiment classification.

Two feature representations are compared in this project: word-level TF-IDF (unigrams +
bigrams) for a simple, fast Logistic Regression baseline, and a word+char TF-IDF union (adds
character n-grams, which are robust to misspellings/typos common in review text) for the
strongest classical candidate. Both operate ONLY on `model_text` -- see src/nlp/amazon/data.py.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_tfidf(
    max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2), min_df: int = 3
) -> TfidfVectorizer:
    """Word-level TF-IDF vectorizer: unigrams+bigrams, English stopwords removed, sublinear TF."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
    )


def build_word_char_union(
    word_max_features: int = 40_000, char_max_features: int = 30_000
) -> FeatureUnion:
    """Word(1-2gram) + char(3-5gram, whitespace-aware) TF-IDF FeatureUnion.

    Character n-grams help with typos, elongated words ("gooood"), and morphological variants
    that word n-grams miss; combined with word n-grams this is the strongest classical
    representation compared in this project.
    """
    word = TfidfVectorizer(
        max_features=word_max_features,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        max_features=char_max_features,
        ngram_range=(3, 5),
        min_df=3,
        analyzer="char_wb",
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
