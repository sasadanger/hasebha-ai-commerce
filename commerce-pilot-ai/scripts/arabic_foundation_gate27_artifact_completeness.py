"""Gate 27: required-artifact completeness audit. Checks that every artifact category listed in
the task brief exists on disk, and reports which (if any) are missing/optional-skipped-with-reason.

Run:
  .venv/Scripts/python.exe scripts/arabic_foundation_gate27_artifact_completeness.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "generated" / "arabic_foundation"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "experiments" / "arabic_foundation"

REQUIRED = {
    "raw_audit": REPORTS_DIR / "labr_full_audit.json",
    "cleaned_manifest_and_exclusion_report": REPORTS_DIR / "split_manifest.json",
    "duplicate_near_dup_report": REPORTS_DIR / "labr_full_audit.json",
    "language_script_report": REPORTS_DIR / "labr_full_audit.json",
    "label_distribution": REPORTS_DIR / "split_manifest.json",
    "split_manifests_dual_hashed": REPORTS_DIR / "split_manifest.json",
    "token_length_audit": REPORTS_DIR / "token_length_audit.json",
    "hardware_audit": REPORTS_DIR / "hardware_audit.json",
    "baseline_model": ARTIFACT_DIR / "baseline" / "tfidf_wordchar_linearsvc.joblib",
    "baseline_predictions": ARTIFACT_DIR / "baseline" / "predictions" / "test_natural_predictions.parquet",
    "transformer_model": ARTIFACT_DIR / "primary_model" / "final" / "model.safetensors",
    "transformer_tokenizer": ARTIFACT_DIR / "primary_model" / "final" / "tokenizer.json",
    "label_mapping": ARTIFACT_DIR / "primary_model" / "final" / "label_mapping.json",
    "training_config": ARTIFACT_DIR / "primary_model" / "final" / "training_config.json",
    "raw_logits": ARTIFACT_DIR / "primary_model" / "final" / "val_natural_logits.npy",
    "calibration_object": ARTIFACT_DIR / "primary_model" / "final" / "calibration.json",
    "calibrated_predictions": ARTIFACT_DIR / "primary_model" / "predictions" / "test_natural_predictions.parquet",
    "final_metrics": REPORTS_DIR / "final_test_evaluation.json",
    "paired_comparison_bootstrap": REPORTS_DIR / "statistical_significance.json",
    "error_analysis_sample": REPORTS_DIR / "error_analysis.json",
    "model_card": REPORTS_DIR / "ARABIC_FOUNDATION_MODEL_CARD.md",
    "reproducibility_manifest": REPORTS_DIR / "ARABIC_FOUNDATION_REPRODUCIBILITY_MANIFEST.md",
    "final_modeling_report": REPORTS_DIR / "ARABIC_FOUNDATION_FINAL_MODELING_REPORT.md",
    "executed_notebook": REPO_ROOT / "notebooks" / "05_arabic_foundation_sentiment_modeling.ipynb",
    "inference_module": REPO_ROOT / "src" / "nlp" / "arabic_foundation" / "inference.py",
    "tests": REPO_ROOT / "tests" / "test_arabic_foundation.py",
    "protected_test_ledger": REPORTS_DIR / "protected_test_access_ledger.json",
}

OPTIONAL_WITH_REASON = {
    "challenger_model": (ARTIFACT_DIR / "challenger" / "final" / "model.safetensors", "Gate 17 CAMeLBERT-Mix challenger -- run only if it demonstrates a distinct question and comparable compute budget"),
    "hard_transfer": (None, "Gate 18 -- HARD dataset confirmed unavailable this session, skipped"),
    "astd_transfer": (None, "Gate 19 -- ASTD intermediate-transfer pilot, optional, run only with prior evidence of benefit"),
    "ordinal_secondary": (None, "Gate 20 -- secondary ordinal 5-class model, optional research-only, may be skipped under time budget"),
    "learning_curve_plot": (None, "not produced as a separate plot artifact -- epoch history is recorded numerically in marbert_training_manifest.json instead"),
}


def main() -> None:
    report = {"required": {}, "optional": {}}
    n_missing = 0
    for name, path in REQUIRED.items():
        exists = path.exists()
        report["required"][name] = {"path": str(path.relative_to(REPO_ROOT)), "exists": exists}
        if not exists:
            n_missing += 1

    for name, (path, reason) in OPTIONAL_WITH_REASON.items():
        exists = path.exists() if path else False
        report["optional"][name] = {"path": str(path.relative_to(REPO_ROOT)) if path else None, "exists": exists, "reason_if_skipped": reason}

    report["n_required_missing"] = n_missing
    report["all_required_present"] = n_missing == 0

    (REPORTS_DIR / "gate27_artifact_completeness.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
