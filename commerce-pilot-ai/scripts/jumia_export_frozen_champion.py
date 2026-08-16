"""Phase 9 -- deterministic export of the frozen Jumia champion (MARBERT,
representative seed 303) from its D: scratch-cache checkpoint into a
durable, hashed artifact location. Copies already-fine-tuned weights only
-- no training performed here, mirroring scripts/nlp_export_inference_artifacts.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_CHECKPOINT = Path("D:/commercepilot_ml_cache/checkpoints/jumia_from_labr_MARBERT_seed202/checkpoint-167")
TOKENIZER_SOURCE = REPO_ROOT / "artifacts" / "experiments" / "nlp" / "inference_exports" / "C_MARBERT"
DEST_DIR = REPO_ROOT / "artifacts" / "experiments" / "jumia" / "phase2_champion" / "JUMIA_MARBERT_LABR_INIT_seed202"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for fname in ["config.json", "model.safetensors"]:
        src = SRC_CHECKPOINT / fname
        dst = DEST_DIR / fname
        shutil.copy2(src, dst)
        hashes[fname] = sha256_file(dst)

    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        src = TOKENIZER_SOURCE / fname
        dst = DEST_DIR / fname
        shutil.copy2(src, dst)
        hashes[fname] = sha256_file(dst)

    manifest = {
        "schema_version": "jumia-champion-export-manifest-v1",
        "exported_at": "2026-08-15",
        "model_name": "UBC-NLP/MARBERT",
        "base_revision": "88e1fa192dd723cf0b3563500aec46209762eb22",
        "initialization": "LABR_FINETUNED_MARBERT (continued fine-tuning from the already-frozen C_MARBERT LABR checkpoint, identical label space, classification head reused directly)",
        "labr_source_checkpoint": "artifacts/experiments/nlp/inference_exports/C_MARBERT",
        "training_seed": 202,
        "max_length": 256,
        "labels": ["1", "2", "3", "4", "5"],
        "source_checkpoint": str(SRC_CHECKPOINT),
        "source_confirmation_result": "reports/generated/jumia/transformer_adaptation/jumia_from_labr_MARBERT_seed202.json",
        "remediation_checkpoint": "reports/checkpoints/jumia_class2_remediation_2026-08-15/CURRENT_STATE.md",
        "dev_macro_f1_evidence_seed202": 0.41242418493257105,
        "dev_macro_f1_evidence_3seed_mean": 0.4120,
        "dev_class2_f1_evidence_seed202": 0.1176,
        "export_dir": str(DEST_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "file_hashes": hashes,
        "not_retrained": True,
        "no_test_access_during_export": True,
    }
    manifest_path = DEST_DIR / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
