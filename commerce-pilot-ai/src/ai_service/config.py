"""Artifact paths and service metadata for the CommercePilot AI service.

Paths are resolved relative to the repository root so the service works the
same whether started from the repo root or another working directory.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVICE_NAME = "commercepilot-ai-service"
SERVICE_VERSION = "0.1.0"

OLIST_MODEL_PATH = (
    REPO_ROOT
    / "artifacts/experiments/olist/phase2a/olist-phase2a-strict-core-v1/models/catboost.cbm"
)
OLIST_MODEL_SHA256 = "5a08ea55332332550a4436f87de91b479fab770a08ec232391d2141bc28a3b2c"
OLIST_EXPERIMENT_ID = "olist-phase2a-strict-core-v1"
# research_threshold from reports/checkpoints/olist_integration_audit_2026-08-11/
# CURRENT_STATE.md (recall 0.79 / precision 0.145 on the frozen test split) --
# reused as-is, not re-derived here.
OLIST_RISK_THRESHOLD = 0.1293
OLIST_RISK_THRESHOLD_SOURCE = "reports/checkpoints/olist_integration_audit_2026-08-11/CURRENT_STATE.md (research_threshold)"

DECISION_ENGINE_CONFIG_PATH = REPO_ROOT / "configs/decision_engine_rules.yaml"

NLP_CHAMPION_REGISTRY_PATH = REPO_ROOT / "configs/nlp_champion_registry.yaml"
NLP_PROMOTION_REGISTRY_PATH = (
    REPO_ROOT
    / "reports/checkpoints/nlp_deployment_promotion_registry_v1_2026-08-14/nlp_deployment_promotion_registry_v1.json"
)
NLP_EXPORT_MANIFEST_PATH = REPO_ROOT / "artifacts/experiments/nlp/inference_exports/export_manifest.json"
NLP_PREPROCESSING_VERSION = "nlp-text-normalization-contract-v2"

INSTACART_POPULARITY_ARTIFACT_PATH = (
    REPO_ROOT / "artifacts/experiments/instacart/phase1_split/frozen_popularity_rank.json"
)
INSTACART_POPULARITY_ARTIFACT_SHA256 = "17ef7f19c8c47375b7b5f4be8803904cb3e04432c02f9e01b62bf9ec6af424dc"
# The frozen ranking logic itself is imported directly from this script (not
# duplicated) so the service can never silently drift from the evidenced/
# frozen candidate; hash-checked at load time against the value recorded in
# reports/checkpoints/instacart_phase1_recommender_freeze_2026-08-14/PRE_TEST_RELEASE_RECORD.md
INSTACART_RECSYS_LIB_PATH = REPO_ROOT / "scripts/instacart_recsys_lib.py"
INSTACART_RECSYS_LIB_SHA256 = "ee32691ec69cfd268ce1e56478fd49f296d6868b063ec75091bb2d9eeff8f0f6"
INSTACART_RECOMMENDER_VERSION = "hybrid_with_popularity_backfill-v1"
INSTACART_FROZEN_K = 10
