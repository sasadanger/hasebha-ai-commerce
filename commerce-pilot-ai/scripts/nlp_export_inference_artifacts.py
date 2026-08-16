"""Deterministic export/cache step for the four NLP inference artifacts the
CommercePilot v1 promotion registry approved (see
reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/).

This does NOT train or fine-tune anything. It copies already-fine-tuned
checkpoint weights (produced by scripts/run_transformer_confirmation.py in an
earlier, already-evidenced session) from the D: scratch-cache location they
were originally written to, into a durable, hashed location under
artifacts/, and saves each checkpoint's tokenizer alongside it (the Trainer
checkpoint dirs only contain model weights + config, not the tokenizer, so
the tokenizer is re-fetched from the pinned-revision HF cache -- also not
training, just reading already-cached files by revision hash).

Source checkpoints (representative seeds, matching
reports/generated/nlp/transformer_confirmation/aggregate_and_bootstrap_summary.json
-> representative_seeds):
  E  (MPOLD) UBC-NLP/MARBERT                       seed 303
  B2 (ASTD)  UBC-NLP/MARBERT                        seed 303
  C  (LABR)  UBC-NLP/MARBERT                        seed 101
  C  (LABR)  aubmindlab/bert-base-arabertv2          seed 202
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/commercepilot_ml_cache/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/commercepilot_ml_cache/hf/hub")
os.environ.setdefault("TORCH_HOME", "D:/commercepilot_ml_cache/torch")
os.environ.setdefault("TMPDIR", "D:/commercepilot_ml_cache/tmp")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = REPO_ROOT / "artifacts" / "experiments" / "nlp" / "inference_exports"
CONFIRMATION_RESULTS_DIR = REPO_ROOT / "reports" / "generated" / "nlp" / "transformer_confirmation"
CHECKPOINTS_ROOT = Path("D:/commercepilot_ml_cache/checkpoints")

EXPORTS = [
    {
        "key": "E_MARBERT",
        "task": "E",
        "task_name": "MPOLD",
        "model_name": "UBC-NLP/MARBERT",
        "revision": "88e1fa192dd723cf0b3563500aec46209762eb22",
        "seed": 303,
        "checkpoint_run": "confirm_E_UBC-NLP__MARBERT_seed303",
    },
    {
        "key": "B2_MARBERT",
        "task": "B2",
        "task_name": "ASTD",
        "model_name": "UBC-NLP/MARBERT",
        "revision": "88e1fa192dd723cf0b3563500aec46209762eb22",
        "seed": 303,
        "checkpoint_run": "confirm_B2_UBC-NLP__MARBERT_seed303",
    },
    {
        "key": "C_MARBERT",
        "task": "C",
        "task_name": "LABR",
        "model_name": "UBC-NLP/MARBERT",
        "revision": "88e1fa192dd723cf0b3563500aec46209762eb22",
        "seed": 101,
        "checkpoint_run": "confirm_C_UBC-NLP__MARBERT_seed101",
    },
    {
        "key": "C_AraBERT",
        "task": "C",
        "task_name": "LABR",
        "model_name": "aubmindlab/bert-base-arabertv2",
        "revision": "97522efce17efa33036ac619802d5cec238dcad9",
        "seed": 202,
        "checkpoint_run": "confirm_C_aubmindlab__bert-base-arabertv2_seed202",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_checkpoint_dir(run_name: str) -> Path:
    run_dir = CHECKPOINTS_ROOT / run_name
    subdirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if len(subdirs) != 1:
        raise RuntimeError(f"expected exactly one checkpoint-* dir under {run_dir}, found {len(subdirs)}")
    return subdirs[0]


def load_confirmation_result(run_name: str) -> dict:
    path = CONFIRMATION_RESULTS_DIR / f"{run_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def export_one(spec: dict) -> dict:
    from transformers import AutoTokenizer

    src_checkpoint = find_checkpoint_dir(spec["checkpoint_run"])
    conf = load_confirmation_result(spec["checkpoint_run"])

    dest_dir = EXPORT_ROOT / spec["key"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["config.json", "model.safetensors"]
    copied_hashes = {}
    for fname in files_to_copy:
        src = src_checkpoint / fname
        dst = dest_dir / fname
        shutil.copy2(src, dst)
        copied_hashes[fname] = sha256_file(dst)

    print(f"[{spec['key']}] loading tokenizer {spec['model_name']}@{spec['revision']} (offline, cached)...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_name"], revision=spec["revision"], cache_dir="D:/commercepilot_ml_cache/hf/hub"
    )
    tokenizer.save_pretrained(str(dest_dir))

    tokenizer_hashes = {}
    for f in sorted(dest_dir.iterdir()):
        if f.name not in copied_hashes:
            tokenizer_hashes[f.name] = sha256_file(f)

    manifest_entry = {
        "key": spec["key"],
        "task": spec["task"],
        "task_name": spec["task_name"],
        "model_name": spec["model_name"],
        "revision": spec["revision"],
        "training_seed": spec["seed"],
        "source_checkpoint": str(src_checkpoint),
        "source_confirmation_result": f"reports/generated/nlp/transformer_confirmation/{spec['checkpoint_run']}.json",
        "dev_macro_f1_evidence": conf["macro_f1"],
        "max_length": conf["max_length"],
        "labels": conf["labels"],
        "export_dir": str(dest_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_file_hashes": copied_hashes,
        "tokenizer_file_hashes": tokenizer_hashes,
        "preprocessing": "src.nlp.text_normalization.normalize_text (nlp-text-normalization-contract-v2), then tokenizer(truncation=True, padding=True, max_length=<max_length>)",
        "not_retrained": True,
        "no_test_access": True,
    }
    print(f"[{spec['key']}] exported to {dest_dir}", file=sys.stderr)
    return manifest_entry


def main() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "nlp-inference-export-manifest-v1",
        "exported_at": "2026-08-15",
        "source": "reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/",
        "note": "Copies already-fine-tuned checkpoint weights only. No training/fine-tuning performed in this script.",
        "exports": [],
    }
    for spec in EXPORTS:
        manifest["exports"].append(export_one(spec))

    manifest_path = EXPORT_ROOT / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
