"""Deterministic Batch 1 configuration resolution and constructor checks."""
from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

FAMILY_CLASS = {
    "TFIDF_LOGISTIC_REGRESSION": LogisticRegression,
    "TFIDF_LINEAR_SVM": LinearSVC,
    "MULTINOMIAL_NAIVE_BAYES": MultinomialNB,
}
FAMILY_DEFAULT_KEY = {
    "TFIDF_LOGISTIC_REGRESSION": "logistic_regression",
    "TFIDF_LINEAR_SVM": "linear_svm",
    "MULTINOMIAL_NAIVE_BAYES": "multinomial_naive_bayes",
}
TOP_LEVEL_KEYS = {"configuration_id", "model_family", "vectorizer", "classifier", "seed"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_batch1_configurations(path: Path) -> list[dict[str, Any]]:
    authorization = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = authorization["execution_defaults"]
    resolved: list[dict[str, Any]] = []
    compound_ids: set[str] = set()
    for experiment_order, experiment in enumerate(authorization["BATCH_1_CANDIDATE"]):
        for configuration_order, declared in enumerate(experiment["learned_configurations"]):
            unknown = set(declared) - TOP_LEVEL_KEYS
            if unknown:
                raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
            family = declared["model_family"]
            if family not in FAMILY_CLASS:
                raise ValueError(f"unsupported model family: {family}")
            if not isinstance(declared.get("configuration_id"), str) or not isinstance(declared.get("seed"), int):
                raise TypeError("configuration_id must be str and seed must be int")
            if not isinstance(declared.get("vectorizer"), dict) or not isinstance(declared.get("classifier"), dict):
                raise TypeError("vectorizer and classifier overrides must be mappings")
            compound_id = f"{experiment['experiment_id']}::{declared['configuration_id']}"
            if compound_id in compound_ids:
                raise ValueError(f"duplicate configuration id: {compound_id}")
            compound_ids.add(compound_id)
            vectorizer = {**deepcopy(defaults["tfidf_vectorizer"]), **deepcopy(declared["vectorizer"])}
            classifier = {**deepcopy(defaults[FAMILY_DEFAULT_KEY[family]]), **deepcopy(declared["classifier"])}
            vectorizer_allowed = set(inspect.signature(TfidfVectorizer).parameters)
            classifier_allowed = set(inspect.signature(FAMILY_CLASS[family]).parameters)
            unknown_vectorizer = set(vectorizer) - vectorizer_allowed
            unknown_classifier = set(classifier) - classifier_allowed
            if unknown_vectorizer or unknown_classifier:
                raise ValueError(
                    f"unknown executable parameters: vectorizer={sorted(unknown_vectorizer)}, "
                    f"classifier={sorted(unknown_classifier)}"
                )
            item = {
                "experiment_id": experiment["experiment_id"],
                "canonical_experiment_id": experiment["canonical_id"],
                "configuration_id": declared["configuration_id"],
                "compound_id": compound_id,
                "model_family": family,
                "vectorizer": vectorizer,
                "classifier": classifier,
                "seed": declared["seed"],
                "execution_order": len(resolved),
                "experiment_order": experiment_order,
                "configuration_order": configuration_order,
            }
            fingerprint_payload = {key: item[key] for key in item if key not in {"execution_order"}}
            item["fingerprint_sha256"] = hashlib.sha256(_canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()
            resolved.append(item)
    return resolved


def instantiate_configuration(configuration: dict[str, Any]) -> tuple[TfidfVectorizer, object]:
    vectorizer_params = deepcopy(configuration["vectorizer"])
    if vectorizer_params.get("dtype") == "float64":
        vectorizer_params["dtype"] = np.float64
    if isinstance(vectorizer_params.get("ngram_range"), list):
        # YAML has no tuple type, so ngram_range arrives as a list (e.g. [1, 2]);
        # TfidfVectorizer's fit-time parameter validation requires a literal
        # tuple instance and rejects a list with the same values. This does not
        # change the resolved configuration or its fingerprint (both are
        # computed earlier, from the list form); it only affects the object
        # actually constructed here.
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    vectorizer = TfidfVectorizer(**vectorizer_params)
    estimator = FAMILY_CLASS[configuration["model_family"]](**deepcopy(configuration["classifier"]))
    return vectorizer, estimator
