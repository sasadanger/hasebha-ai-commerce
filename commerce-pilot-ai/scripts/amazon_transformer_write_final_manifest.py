"""Gate 9: consolidate all transformer-pipeline artifacts, hashes, and results into
reports/generated/amazon/transformer_metrics.json (parallel to, never overwriting, metrics.json).

Run once from repo root, after all prior transformer scripts have completed:
  .venv/Scripts/python.exe scripts/amazon_transformer_write_final_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "amazon"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "experiments" / "amazon" / "transformer"
SPLIT_IDS_DIR = REPORTS_DIR / "split_ids"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_of_uids(path: Path) -> str:
    import pandas as pd

    df = pd.read_parquet(path)
    joined = "\n".join(sorted(df["review_uid"].astype(str).tolist()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> None:
    hardware = json.loads((REPORTS_DIR / "transformer_hardware_audit.json").read_text())
    data_audit = json.loads((REPORTS_DIR / "transformer_data_audit.json").read_text())
    split_manifest = json.loads((REPORTS_DIR / "transformer_split_manifest.json").read_text())
    token_audit = json.loads((REPORTS_DIR / "transformer_token_length_audit.json").read_text())
    smoke_test = json.loads((REPORTS_DIR / "transformer_smoke_test.json").read_text())
    training_run = json.loads((REPORTS_DIR / "transformer_training_run.json").read_text())
    calibration = json.loads((REPORTS_DIR / "transformer_calibration.json").read_text())
    final_eval = json.loads((REPORTS_DIR / "transformer_final_eval.json").read_text())
    model_manifest = json.loads((ARTIFACTS_DIR / "model_manifest.json").read_text())
    tfidf_metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())

    artifact_hashes = {
        "model.safetensors": sha256_file(ARTIFACTS_DIR / "model" / "model.safetensors"),
        "config.json": sha256_file(ARTIFACTS_DIR / "model" / "config.json"),
        "tokenizer.json": sha256_file(ARTIFACTS_DIR / "model" / "tokenizer.json"),
        "tokenizer_config.json": sha256_file(ARTIFACTS_DIR / "model" / "tokenizer_config.json"),
    }

    split_manifest_hashes = {
        "split_manifest.json (TF-IDF splits, historical, unmodified)": sha256_file(REPORTS_DIR / "split_manifest.json"),
        "transformer_split_manifest.json (val_natural)": sha256_file(REPORTS_DIR / "transformer_split_manifest.json"),
    }
    for name in [
        "val", "val_natural", "test_balanced", "test_representative",
        "product_holdout_stress", "chronological_stress", "train_full_pool",
        "learning_curve_25000", "learning_curve_50000", "learning_curve_100000", "learning_curve_200000",
    ]:
        p = SPLIT_IDS_DIR / f"{name}.parquet"
        if p.exists():
            split_manifest_hashes[f"split_ids/{name}.parquet"] = sha256_of_uids(p)

    prediction_hashes = {}
    for p in (REPORTS_DIR / "predictions").glob("*.parquet"):
        prediction_hashes[p.name] = sha256_file(p)

    out = {
        "generated_at": "2026-08-17",
        "summary": {
            "primary_model_recommendation": (
                "transformer" if final_eval["overall_verdict"]["eval_sets_transformer_wins_macro_f1"]
                == final_eval["overall_verdict"]["eval_sets_total"] else "mixed_or_tfidf_still_competitive"
            ),
            "checkpoint": training_run["checkpoint"],
            "used_fallback_checkpoint": training_run["used_fallback_checkpoint"],
            "training_size": training_run["training_size"],
            "expanded_to_100k": training_run["hundred_k_expansion_decision"]["DECISION_expand_to_100k"],
            "macro_f1_deltas_vs_tfidf_by_eval_set": final_eval["overall_verdict"]["macro_f1_deltas_by_set"],
            "mean_macro_f1_delta_vs_tfidf": final_eval["overall_verdict"]["mean_delta"],
            "transformer_wins_on_n_of_4_eval_sets": final_eval["overall_verdict"]["eval_sets_transformer_wins_macro_f1"],
        },
        "gate1_hardware_audit": hardware,
        "gate2_data_audit_summary": {
            "file": "reports/generated/amazon/transformer_data_audit.json",
            "audit_sample_size": data_audit["audit_sample"]["size"],
        },
        "gate3_new_split": {
            "file": "reports/generated/amazon/transformer_split_manifest.json",
            "val_natural_achieved_counts": split_manifest["val_natural_achieved_counts"],
            "disjointness_report": split_manifest["disjointness_report"],
        },
        "gate4_token_length_and_max_length_decision": token_audit["decision"],
        "gate4_token_length_percentiles": token_audit["token_length_percentiles_full_50k_no_truncation"],
        "gate5_smoke_test": {
            k: v for k, v in smoke_test.items() if k != "train_loss_sequence"
        },
        "gate6_training_run": {
            "checkpoint": training_run["checkpoint"],
            "train_config": training_run["train_config"],
            "hardware": training_run["hardware"],
            "per_epoch_eval_on_balanced_val": training_run["per_epoch_eval_on_balanced_val"],
            "per_epoch_eval_on_val_natural_reporting_only": training_run[
                "per_epoch_eval_on_val_natural_REPORTING_ONLY_not_used_for_any_decision"
            ],
            "epoch_decision": training_run["epoch_decision"],
            "hundred_k_expansion_decision": training_run["hundred_k_expansion_decision"],
        },
        "gate7_calibration": calibration,
        "gate8_final_evaluation": final_eval["results"],
        "gate8_overall_verdict": final_eval["overall_verdict"],
        "gate8_frozen_tfidf_baseline_for_reference": {
            name: tfidf_metrics["results"][name]["final_model"]["macro_f1"]
            for name in ["test_balanced", "test_representative", "product_holdout_stress", "chronological_stress"]
        },
        "artifact_hashes": {
            "model_files": artifact_hashes,
            "model_manifest": model_manifest,
            "split_manifests_and_split_ids": split_manifest_hashes,
            "predictions": prediction_hashes,
        },
        "artifact_paths": {
            "checkpoint_dir": "artifacts/experiments/amazon/transformer/model/",
            "model_manifest": "artifacts/experiments/amazon/transformer/model_manifest.json",
            "tokenized_cache_dir": "artifacts/experiments/amazon/transformer/tokenized_cache/",
            "predictions_dir": "reports/generated/amazon/predictions/ (*_transformer_predictions.parquet, val_natural_transformer_predictions.parquet)",
            "hardware_audit": "reports/generated/amazon/transformer_hardware_audit.json",
            "data_audit": "reports/generated/amazon/transformer_data_audit.json",
            "token_length_audit": "reports/generated/amazon/transformer_token_length_audit.json",
            "smoke_test": "reports/generated/amazon/transformer_smoke_test.json",
            "training_run": "reports/generated/amazon/transformer_training_run.json",
            "calibration": "reports/generated/amazon/transformer_calibration.json",
            "final_eval": "reports/generated/amazon/transformer_final_eval.json",
            "split_manifest_new_bucket": "reports/generated/amazon/transformer_split_manifest.json",
            "val_natural_split_ids": "reports/generated/amazon/split_ids/val_natural.parquet",
        },
    }
    (REPORTS_DIR / "transformer_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote reports/generated/amazon/transformer_metrics.json")


if __name__ == "__main__":
    main()
