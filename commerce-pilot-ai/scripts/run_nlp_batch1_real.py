"""Execute one real Phase 2C NLP Batch 1 experiment (A/B2/C/E) end to end.

This is the "future authorized agent" entry point the V5 executor architecture
(src/nlp/batch1_executor.py) was built for: it sources real, hash-verified
acquisition data, applies the authorized schema adapter where the manifest
declares one, and calls execute_batch1_experiment with the real resolved
20-configuration set from configs/nlp_training_batch_authorization_v2.yaml.

It stops at validated winner selection and artifact writing. It never calls
release_control -- internal-test release remains a separately authorized step.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_nlp_batch1_real.py A
    .venv\\Scripts\\python.exe scripts\\run_nlp_batch1_real.py B2
    .venv\\Scripts\\python.exe scripts\\run_nlp_batch1_real.py C
    .venv\\Scripts\\python.exe scripts\\run_nlp_batch1_real.py E
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp.amazon_adapter import adapt_amazon_record
from src.nlp.batch1_executor import Batch1ExecutorInputs, execute_batch1_experiment
from src.nlp.configuration import _canonical_json, instantiate_configuration, resolve_batch1_configurations

AUTH_PATH = ROOT / "configs" / "nlp_training_batch_authorization_v2.yaml"
ACQUISITION_MANIFEST_PATH = ROOT / "reports" / "generated" / "nlp" / "acquisition_manifest_v2.json"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expected_hash(dataset_id: str, filename: str) -> str:
    manifest = json.loads(ACQUISITION_MANIFEST_PATH.read_text(encoding="utf-8"))
    for dataset in manifest["datasets"]:
        if dataset["dataset_id"] == dataset_id:
            files = dataset["files"]
            if isinstance(files, dict):
                return files[filename]
    raise KeyError(f"no acquisition manifest entry for {dataset_id}/{filename}")


def load_amazon_records() -> tuple[list[dict], str, str]:
    source = ROOT / "data" / "raw" / "amazon_reviews_appliances" / "Appliances.jsonl.gz"
    actual_sha256 = sha256_of_file(source)
    expected_sha256 = "150f209befceaa6f837abc997065b2d251034bbbda19bebc4ad56dac779730c2"
    con = duckdb.connect()
    relation = f"read_json_auto('{source.as_posix()}', format='newline_delimited', maximum_object_size=33554432)"
    rows = con.execute(f"SELECT rating, title, text FROM {relation}").fetchall()
    con.close()
    records = [{"rating": r[0], "title": r[1], "text": r[2]} for r in rows]
    return records, actual_sha256, expected_sha256


def load_astd_records() -> tuple[list[dict], str, str]:
    source = ROOT / "data" / "quarantine" / "nlp" / "astd" / "data_Tweets.txt"
    actual_sha256 = sha256_of_file(source)
    expected_sha256 = _expected_hash("astd", "data_Tweets.txt")
    records = []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"unexpected ASTD row shape: {parts!r}")
            text, label = parts
            records.append({"text": text, "label": label})
    return records, actual_sha256, expected_sha256


def load_labr_records() -> tuple[list[dict], str, str]:
    source = ROOT / "data" / "quarantine" / "nlp" / "labr" / "reviews.tsv"
    actual_sha256 = sha256_of_file(source)
    expected_sha256 = _expected_hash("labr", "reviews.tsv")
    records = []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f"unexpected LABR row shape: {parts!r}")
            rating, review_id, user_id, book_id, review_text = parts
            records.append({"rating": int(rating), "review_text": review_text})
    return records, actual_sha256, expected_sha256


def load_mpold_records() -> tuple[list[dict], str, str]:
    source = ROOT / "data" / "quarantine" / "nlp" / "mpold" / "Arabic_offensive_comment_detection_annotation_4000_selected.xlsx"
    actual_sha256 = sha256_of_file(source)
    expected_sha256 = _expected_hash("mpold", source.name)
    df = pd.read_excel(source)
    records = [
        {"Comment": row["Comment"], "Majority_Label": row["Majority_Label"]}
        for _, row in df.iterrows()
    ]
    return records, actual_sha256, expected_sha256


EXPERIMENT_SPECS = {
    "A": {
        "task_type": "FIVE_CLASS_RATING_CLASSIFICATION",
        "text_key": "review_text", "label_key": "overall",
        "loader": load_amazon_records, "schema_adapter": adapt_amazon_record,
    },
    "B2": {
        "task_type": "FOUR_CLASS_SENTIMENT_CLASSIFICATION",
        "text_key": "text", "label_key": "label",
        "loader": load_astd_records, "schema_adapter": None,
    },
    "C": {
        "task_type": "FIVE_CLASS_RATING_CLASSIFICATION",
        "text_key": "review_text", "label_key": "rating",
        "loader": load_labr_records, "schema_adapter": None,
    },
    "E": {
        "task_type": "BINARY_OFFENSIVE_LANGUAGE_CLASSIFICATION",
        "text_key": "Comment", "label_key": "Majority_Label",
        "loader": load_mpold_records, "schema_adapter": None,
    },
}


def real_metric_function(y_true, y_pred):
    # Deliberately does NOT pass zero_division: nlp_metric_contract_v2.yaml does
    # not pin a zero-division policy (documented residual ambiguity in the V5
    # remediation summary, section 5). Rather than invent a project-specific
    # rule, this defers entirely to scikit-learn's own library default.
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def build_run_id(configurations: list[dict]) -> str:
    payload = sorted(c["fingerprint_sha256"] for c in configurations)
    config_set_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"nlp-batch1-{timestamp}-{config_set_hash}"


def run(experiment_id: str) -> None:
    if experiment_id not in EXPERIMENT_SPECS:
        raise SystemExit(f"unknown experiment_id: {experiment_id!r}; must be one of {sorted(EXPERIMENT_SPECS)}")
    spec = EXPERIMENT_SPECS[experiment_id]

    print(f"[{experiment_id}] loading real acquisition data...", flush=True)
    records, actual_sha256, expected_sha256 = spec["loader"]()
    print(f"[{experiment_id}] loaded {len(records)} rows; acquisition sha256 actual={actual_sha256[:12]}... expected={expected_sha256[:12]}...", flush=True)
    if actual_sha256 != expected_sha256:
        raise SystemExit(f"[{experiment_id}] ACQUISITION HASH MISMATCH: refusing to proceed")

    all_configurations = resolve_batch1_configurations(AUTH_PATH)
    configurations = [c for c in all_configurations if c["experiment_id"] == experiment_id]
    print(f"[{experiment_id}] resolved {len(configurations)} configurations: {[c['compound_id'] for c in configurations]}", flush=True)

    run_id = build_run_id(configurations)
    print(f"[{experiment_id}] run_id={run_id}", flush=True)

    inputs = Batch1ExecutorInputs(
        run_id=run_id, experiment_id=experiment_id, task_type=spec["task_type"],
        canonical_records=records, text_key=spec["text_key"], label_key=spec["label_key"],
        seed=20260809, resolved_configurations=configurations,
        estimator_factory=instantiate_configuration, metric_function=real_metric_function,
        dataset_acquisition_sha256=actual_sha256, expected_dataset_acquisition_sha256=expected_sha256,
        schema_adapter=spec["schema_adapter"],
    )

    print(f"[{experiment_id}] executing (train-only fit, validation-only eval, no internal-test access)...", flush=True)
    result = execute_batch1_experiment(inputs)

    print(f"[{experiment_id}] DONE. winner={result.winner.compound_id} split_hash={result.split_hash[:12]}...", flush=True)
    for row in result.results:
        print(f"    {row['compound_id']}: macro_f1={row['macro_f1']:.4f} balanced_accuracy={row['balanced_accuracy']:.4f} accuracy={row['accuracy']:.4f}", flush=True)
    print(f"[{experiment_id}] audit: {result.prep_result.audit}", flush=True)
    print(f"[{experiment_id}] artifacts written under: {next(iter(result.artifact_paths.values())).parent}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_nlp_batch1_real.py <A|B2|C|E>")
    run(sys.argv[1])
