"""Model training for Amazon Appliances binary sentiment classification.

All real models are scikit-learn Pipelines (vectorizer + classifier) fit ONLY on `model_text`
strings -- never on `rating` or `verified_purchase`. Every model exposes `.predict(texts)` and,
where meaningful, `.predict_proba(texts)` returning P(class) with columns [P(negative),
P(positive)], so evaluate.py can treat all models uniformly. Random seeds are fixed everywhere.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp.amazon.features import build_word_char_union, build_word_tfidf

SEED = 20260817


class DummyStratifiedBaseline:
    """Stratified-random dummy baseline.

    Predicts labels sampled from the TRAINING class distribution, independent of review text.
    Documented choice (over majority-class) because the modeling subsets built by
    src/nlp/amazon/data.py are class-balanced by construction, so a majority-class dummy would
    trivially score 50% accuracy / low macro-F1 on those -- stratified-random gives a slightly
    more informative floor and is the more common convention for this kind of report. Both the
    balanced and representative test sets independently confirm this is a real, weak floor
    (see reports/generated/amazon/metrics.json).
    """

    def __init__(self, random_state: int = SEED) -> None:
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        self.class_prior_: np.ndarray | None = None

    def fit(self, texts, labels) -> "DummyStratifiedBaseline":
        labels = np.asarray(labels)
        self.class_prior_ = np.array([(labels == 0).mean(), (labels == 1).mean()])
        return self

    def predict(self, texts) -> np.ndarray:
        n = len(texts)
        rng = np.random.default_rng(self.random_state)
        return rng.choice([0, 1], size=n, p=self.class_prior_)

    def predict_proba(self, texts) -> np.ndarray:
        n = len(texts)
        return np.tile(self.class_prior_, (n, 1))


def train_dummy(texts, labels) -> DummyStratifiedBaseline:
    """Fit the stratified-random dummy baseline."""
    model = DummyStratifiedBaseline(random_state=SEED)
    model.fit(texts, labels)
    return model


def train_tfidf_logreg(texts, labels) -> Pipeline:
    """Word-level TF-IDF + Logistic Regression."""
    pipe = Pipeline(
        [
            ("tfidf", build_word_tfidf()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def train_tfidf_char_linearsvc(texts, labels) -> Pipeline:
    """Word+char TF-IDF + LinearSVC, calibrated (3-fold sigmoid calibration) so predict_proba is
    available for ROC-AUC / calibration reporting. This is the strongest classical candidate.
    """
    pipe = Pipeline(
        [
            ("tfidf", build_word_char_union()),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(random_state=SEED, dual="auto"), cv=3, method="sigmoid"
                ),
            ),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


MODEL_TRAINERS = {
    "dummy_stratified": train_dummy,
    "tfidf_word_logreg": train_tfidf_logreg,
    "tfidf_wordchar_linearsvc": train_tfidf_char_linearsvc,
}
