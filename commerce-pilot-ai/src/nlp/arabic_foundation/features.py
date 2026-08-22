"""TF-IDF vectorizer construction for the Arabic sentiment foundation classical baseline.

Mirrors the convention established in src/nlp/amazon/features.py (word TF-IDF for a simple
baseline, word+char TF-IDF union for the strongest classical candidate) adapted for Arabic:
no English stopword list is applied (there is no Arabic stopword removal per the task's light-
normalization-only principle -- stopword removal is explicitly disallowed by the brief), and
char n-grams matter even more here because Arabic morphology + dialectal spelling variation +
elongation ("رااااائع") are common and word n-grams alone miss them.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_char_union(
    word_max_features: int = 40_000, char_max_features: int = 40_000
) -> FeatureUnion:
    """Word(1-2gram) + char(2-5gram, whitespace-aware) TF-IDF FeatureUnion for Arabic text.

    No stop-word list (none applied for Arabic; also disallowed by the light-normalization-only
    brief). char_wb n-grams starting at 2 (rather than 3, as in the Amazon English baseline)
    because Arabic words are frequently shorter once affixes are accounted for.
    """
    word = TfidfVectorizer(
        max_features=word_max_features,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        max_features=char_max_features,
        ngram_range=(2, 5),
        min_df=3,
        analyzer="char_wb",
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
